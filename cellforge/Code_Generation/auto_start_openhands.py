#!/usr/bin/env python3
"""
Fully Automated OpenHands Startup Script for CellForge Code Generation
No human intervention required - completely automated startup process
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path
import logging
import signal
import atexit

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('openhands_startup.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AutomatedOpenHands:
    """Fully automated OpenHands startup and management"""
    
    def __init__(self):
        self.openhands_path = Path(__file__).parent / "OpenHands"
        self.container_name = "openhands-app-automated"
        self.is_running = False
        
        # Register cleanup function
        atexit.register(self.cleanup)
        
        # Handle signals for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Cleanup function to stop containers on exit"""
        if self.is_running:
            logger.info("Cleaning up OpenHands containers...")
            self.stop_openhands()
    
    def check_prerequisites(self) -> bool:
        """Check all prerequisites automatically"""
        logger.info("Checking prerequisites...")
        
        # Check Docker
        if not self._check_docker():
            logger.error("Docker is required but not available")
            return False
        
        # Check Docker Compose
        if not self._check_docker_compose():
            logger.error("Docker Compose is required but not available")
            return False
        
        # Check network connectivity
        if not self._check_network_connectivity():
            logger.error("Network connectivity issues detected")
            return False
        
        # Check OpenHands directory
        if not self.openhands_path.exists():
            logger.error(f"OpenHands directory not found: {self.openhands_path}")
            return False
        
        # Check docker-compose.yml
        docker_compose_file = self.openhands_path / "docker-compose.yml"
        if not docker_compose_file.exists():
            logger.error(f"docker-compose.yml not found in: {self.openhands_path}")
            return False
        
        # Check if we have research plan to work with
        research_plan_path = Path("results/research_plan.json")
        if research_plan_path.exists():
            logger.info(f"Found research plan: {research_plan_path}")
        else:
            logger.warning(f"Research plan not found: {research_plan_path}")
            logger.info("OpenHands will start without research plan")
        
        logger.info("All prerequisites satisfied")
        return True
    
    def _check_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"Docker available: {result.stdout.strip()}")
                return True
            else:
                logger.error("Docker not available")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.error("Docker not found or not accessible")
            return False
    
    def _check_docker_compose(self) -> bool:
        """Check if Docker Compose is available"""
        try:
            result = subprocess.run(
                ["docker", "compose", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"Docker Compose available: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"Docker Compose not available: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error checking Docker Compose: {e}")
            return False
    
    def _check_network_connectivity(self) -> bool:
        """Check network connectivity to required services"""
        logger.info("Checking network connectivity...")
        
        # Test basic internet connectivity
        try:
            import urllib.request
            urllib.request.urlopen('https://www.google.com', timeout=10)
            logger.info("Basic internet connectivity: OK")
        except Exception as e:
            logger.error(f"Basic internet connectivity failed: {e}")
            return False
        
        # Test GitHub Container Registry connectivity
        try:
            import urllib.request
            urllib.request.urlopen('https://ghcr.io', timeout=10)
            logger.info("GitHub Container Registry connectivity: OK")
        except Exception as e:
            logger.error(f"GitHub Container Registry connectivity failed: {e}")
            logger.warning("This may cause issues pulling Docker images")
            return False
        
        # Test Docker Hub connectivity
        try:
            import urllib.request
            urllib.request.urlopen('https://registry-1.docker.io', timeout=10)
            logger.info("Docker Hub connectivity: OK")
        except Exception as e:
            logger.error(f"Docker Hub connectivity failed: {e}")
            logger.warning("This may cause issues pulling Docker images")
            return False
        
        return True
    
    def stop_existing_containers(self) -> bool:
        """Stop any existing OpenHands containers"""
        try:
            logger.info("Stopping any existing OpenHands containers...")
            
            # Stop any containers with openhands in the name
            result = subprocess.run(
                ["docker", "ps", "-q", "--filter", "name=openhands"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.stdout.strip():
                container_ids = result.stdout.strip().split('\n')
                for container_id in container_ids:
                    if container_id:
                        logger.info(f"Stopping container: {container_id}")
                        subprocess.run(
                            ["docker", "stop", container_id],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        subprocess.run(
                            ["docker", "rm", container_id],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                logger.info("Existing containers stopped and removed")
            else:
                logger.info("No existing OpenHands containers found")
            
            return True
                
        except Exception as e:
            logger.error(f"Error stopping existing containers: {e}")
            return False
    
    def start_openhands(self) -> bool:
        """Start OpenHands using Docker with retry mechanism"""
        logger.info("Starting OpenHands using official Docker image...")
        
        # Stop any existing containers first
        self.stop_openhands()
        
        # Create workspace directory
        workspace_dir = Path.home() / ".openhands-workspace"
        workspace_dir.mkdir(exist_ok=True)
        
        # Create basic prompt if no research plan exists
        research_plan_path = Path("results/research_plan.json")
        if not research_plan_path.exists():
            logger.warning("No research plan found, creating basic prompt")
            initial_prompt = workspace_dir / "initial_prompt.md"
            with open(initial_prompt, 'w', encoding='utf-8') as f:
                f.write("# OpenHands Initial Prompt\n\nThis is a basic prompt for OpenHands to get started.\n")
            logger.info(f"Created basic prompt: {initial_prompt}")
        
        # Try to pull image first with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to pull OpenHands image (attempt {attempt + 1}/{max_retries})...")
                
                # First try to pull the image
                pull_cmd = [
                    "docker", "pull", "docker.all-hands.dev/all-hands-ai/openhands:0.30"
                ]
                
                pull_result = subprocess.run(
                    pull_cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minutes timeout for pull
                )
                
                if pull_result.returncode == 0:
                    logger.info("Successfully pulled OpenHands image")
                    break
                else:
                    logger.warning(f"Failed to pull image (attempt {attempt + 1}): {pull_result.stderr}")
                    if attempt < max_retries - 1:
                        logger.info("Waiting 30 seconds before retry...")
                        time.sleep(30)
                    else:
                        logger.error("Failed to pull image after all attempts")
                        return False
                        
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout pulling image (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    logger.info("Waiting 30 seconds before retry...")
                    time.sleep(30)
                else:
                    logger.error("Timeout pulling image after all attempts")
                    return False
            except Exception as e:
                logger.error(f"Error pulling image: {e}")
                return False
        
        # Now start the container
        try:
            cmd = [
                "docker", "run", "-d",
                "--name", self.container_name,
                "-p", "3000:3000",
                "-v", "/var/run/docker.sock:/var/run/docker.sock",
                "-v", f"{workspace_dir}:/opt/workspace_base",
                "-v", f"{Path.home()}/.openhands-state:/.openhands-state",
                "--add-host", "host.docker.internal:host-gateway",
                "-e", "SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.30-nikolaik",
                "-e", "LOG_ALL_EVENTS=true",
                "docker.all-hands.dev/all-hands-ai/openhands:0.30"
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("OpenHands container started successfully")
                if result.stdout:
                    logger.info(f"Container ID: {result.stdout.strip()}")
                self.is_running = True
                return True
            else:
                logger.error(f"Failed to start OpenHands: {result.stderr}")
                if result.stdout:
                    logger.error(f"Command output: {result.stdout}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Timeout starting OpenHands container")
            return False
        except Exception as e:
            logger.error(f"Error starting OpenHands: {e}")
            return False
    
    def wait_for_openhands_ready(self, timeout: int = 300) -> bool:
        """Wait for OpenHands to be ready with extended timeout"""
        logger.info("Waiting for OpenHands to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if OpenHands is responding
                response = requests.get("http://localhost:3000", timeout=15)
                if response.status_code == 200:
                    logger.info("OpenHands is ready!")
                    return True
                else:
                    logger.info(f"OpenHands responding with status: {response.status_code}")
            except requests.RequestException as e:
                logger.info(f"OpenHands not ready yet: {e}")
            
            time.sleep(10)  # Check every 10 seconds
        
        logger.error("OpenHands did not become ready within timeout")
        return False
    
    def test_openhands_api(self) -> bool:
        """Test OpenHands API functionality"""
        try:
            logger.info("Testing OpenHands API...")
            
            # Test basic connectivity
            response = requests.get("http://localhost:3000", timeout=15)
            if response.status_code == 200:
                logger.info("OpenHands API is responding")
                
                # Test chat endpoint
                api_url = "http://localhost:3000/api/chat"
                test_payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hello, this is a test message."
                        }
                    ],
                    "model": "anthropic/claude-3-5-sonnet-20241022",
                    "temperature": 0.1,
                    "max_tokens": 100
                }
                
                response = requests.post(
                    api_url,
                    json=test_payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    logger.info("OpenHands chat API is working")
                    return True
                else:
                    logger.error(f"OpenHands chat API error: {response.status_code} - {response.text}")
                    return False
            else:
                logger.error(f"OpenHands API not responding properly: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error testing OpenHands API: {e}")
            return False
    
    def stop_openhands(self) -> bool:
        """Stop OpenHands container"""
        try:
            logger.info("Stopping OpenHands container...")
            
            # Stop and remove container
            result = subprocess.run(
                ["docker", "stop", "openhands-app"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info("OpenHands container stopped successfully")
                
                # Remove container
                subprocess.run(["docker", "rm", "openhands-app"], capture_output=True)
                logger.info("OpenHands container removed")
                
                self.is_running = False
                return True
            else:
                logger.error(f"Failed to stop OpenHands: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error stopping OpenHands: {e}")
            return False
    
    def _create_initial_prompt(self, workspace_dir: Path):
        """Create initial prompt based on research plan"""
        try:
            import json
            
            # Read research plan
            research_plan_path = Path("results/research_plan.json")
            with open(research_plan_path, 'r', encoding='utf-8') as f:
                research_plan = json.load(f)
            
            # Extract key information
            task_description = research_plan.get("task_description", "Single-cell perturbation prediction")
            dataset_info = research_plan.get("dataset", {})
            perturbations = research_plan.get("perturbations", [])
            
            # Create comprehensive prompt
            prompt_content = f"""# CellForge Research Plan Implementation

## Task Description
{task_description}

## Dataset Information
- Dataset: {dataset_info.get('name', 'Unknown')}
- Type: {dataset_info.get('type', 'Single-cell RNA-seq')}
- Cell Types: {', '.join(dataset_info.get('cell_types', []))}

## Perturbations to Study
{chr(10).join([f"- {p.get('type', 'Unknown')}: {', '.join(p.get('targets', []))}" for p in perturbations])}

## Implementation Requirements

Please implement the research plan by:

1. **Data Processing Pipeline**
   - Set up quality control for single-cell data
   - Implement normalization and batch correction
   - Create feature selection pipeline

2. **Model Architecture**
   - Design perturbation prediction model
   - Implement cross-attention mechanism
   - Set up training pipeline

3. **Evaluation Framework**
   - Create evaluation metrics
   - Set up validation pipeline
   - Implement result visualization

4. **Code Organization**
   - Create modular Python code
   - Include proper documentation
   - Add configuration files

## Files to Create
- `data_processing.py` - Data preprocessing pipeline
- `model.py` - Model architecture definition
- `train.py` - Training script
- `evaluate.py` - Evaluation script
- `config.py` - Configuration management
- `requirements.txt` - Dependencies
- `README.md` - Documentation

Please start by examining the research plan and creating the implementation structure.
"""
            
            # Write prompt to workspace
            prompt_file = workspace_dir / "initial_prompt.md"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt_content)
            
            logger.info(f"Created initial prompt: {prompt_file}")
            
        except Exception as e:
            logger.error(f"Error creating initial prompt: {e}")
    
    def _create_basic_prompt(self, workspace_dir: Path):
        """Create basic prompt when no research plan exists"""
        try:
            prompt_content = """# CellForge Code Generation

## Task
Please help me implement a single-cell perturbation prediction system.

## Requirements
1. Create a data processing pipeline for single-cell RNA-seq data
2. Design a deep learning model for perturbation prediction
3. Implement training and evaluation scripts
4. Add proper documentation and configuration

## Files to Create
- `data_processing.py`
- `model.py` 
- `train.py`
- `evaluate.py`
- `config.py`
- `requirements.txt`
- `README.md`

Please start by creating the project structure and implementing the core components.
"""
            
            # Write prompt to workspace
            prompt_file = workspace_dir / "initial_prompt.md"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt_content)
            
            logger.info(f"Created basic prompt: {prompt_file}")
            
        except Exception as e:
            logger.error(f"Error creating basic prompt: {e}")
    
    def start_automated(self) -> bool:
        """Fully automated startup process"""
        logger.info("Starting fully automated OpenHands startup process...")
        
        try:
            # Step 1: Check prerequisites
            if not self.check_prerequisites():
                logger.error("Prerequisites check failed")
                return False
            
            # Step 2: Stop existing containers
            if not self.stop_existing_containers():
                logger.warning("Warning: Failed to stop existing containers")
            
            # Step 3: Start OpenHands
            if not self.start_openhands():
                logger.error("Failed to start OpenHands")
                return False
            
            # Step 4: Wait for OpenHands to be ready
            if not self.wait_for_openhands_ready():
                logger.error("OpenHands did not become ready")
                return False
            
            # Step 5: Test OpenHands API
            if not self.test_openhands_api():
                logger.error("OpenHands API test failed")
                return False
            
            logger.info("OpenHands is ready for use!")
            logger.info("OpenHands is available at: http://localhost:3000")
            logger.info("Log file: openhands_startup.log")
            
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error in automated startup: {e}")
            return False

def main():
    """Main function - completely automated"""
    logger.info("=== Fully Automated OpenHands Startup ===")
    logger.info("No human intervention required")
    
    # Create automated OpenHands instance
    openhands = AutomatedOpenHands()
    
    # Start automated process
    success = openhands.start_automated()
    
    if success:
        print("\nOpenHands startup completed successfully!")
        print("All processes were automated - no human intervention required")
        print("OpenHands is available at: http://localhost:3000")
        print("Check openhands_startup.log for detailed logs")
        
        # Keep the process running
        try:
            logger.info("Keeping OpenHands running... Press Ctrl+C to stop")
            while True:
                time.sleep(60)  # Check every minute
                # Optional: Add health check here
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
            openhands.cleanup()
    else:
        print("\nOpenHands startup failed!")
        print("Check openhands_startup.log for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()
