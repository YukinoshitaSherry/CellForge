#!/usr/bin/env python3
"""
Agentic Coder Integration Module for scAgents-2025
Automates code generation by interacting with a command-line-based agentic tool running in Docker.
"""

import os
import json
import subprocess
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import docker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgenticCoder:
    """
    Manages the interaction with a Docker-based agentic coder CLI to generate
    executable code from Method Design plans.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.docker_client = docker.from_env()
        except docker.errors.DockerException:
            logger.error("Docker is not running or not installed. Please start Docker and try again.")
            self.docker_client = None
        self.container_name = f"agentic-coder-app-{int(time.time())}"
        
    def convert_plan_to_prompt(self, plan_data: Dict[str, Any]) -> str:
        """Converts a Method Design JSON plan into a detailed textual prompt for code generation."""
        # This function generates a detailed prompt based on the structured JSON plan.
        # (The implementation logic is preserved from the original file)
        prompt = f"""
# Single-Cell Analysis Code Generation

Based on the following research plan, generate complete, executable Python code for single-cell data analysis.

## Research Plan Summary:
- **Task Type**: {plan_data.get('report_metadata', {}).get('task_type', 'Unknown')}
- **Dataset**: {plan_data.get('report_metadata', {}).get('dataset', 'Unknown')}
- **Title**: {plan_data.get('report_metadata', {}).get('title', 'Unknown')}

## Data Preprocessing Requirements:
"""
        
        if 'data_preprocessing' in plan_data:
            for step in plan_data['data_preprocessing'].get('steps', []):
                prompt += f"- {step.get('step', '')}: {step.get('description', '')}\n"
                if 'parameters' in step:
                    prompt += f"  Parameters: {step['parameters']}\n"
        
        prompt += "\n## Model Architecture Requirements:\n"
        
        if 'model_design' in plan_data:
            prompt += f"- Overview: {plan_data['model_design'].get('overview', '')}\n"
            for component in plan_data['model_design'].get('key_components', []):
                prompt += f"- {component.get('name', '')}: {component.get('purpose', '')}\n"
                if 'architecture' in component:
                    prompt += f"  Architecture: {component['architecture']}\n"
        
        prompt += "\n## Training Strategy:\n"
        
        if 'training_strategy' in plan_data:
            training = plan_data['training_strategy']
            prompt += f"- Loss Function: {training.get('loss_function', {})}\n"
            prompt += f"- Optimizer: {training.get('optimizer', {})}\n"
            prompt += f"- Training Parameters: {training.get('training_parameters', {})}\n"
        
        prompt += """
## Requirements:
1. Generate complete, executable Python code in the current directory.
2. Use PyTorch for deep learning components and Scanpy for single-cell data processing.
3. Include proper error handling, logging, data validation, and quality checks.
4. Implement the exact architecture and training strategy specified.
5. Add model evaluation, visualization, and configuration management.
6. Provide a `requirements.txt` file with all dependencies.

## Code Structure:
- `main.py`: Main execution script
- `model.py`: Model architecture definition
- `data_loader.py`: Data preprocessing and loading
- `train.py`: Training loop and functions
- `evaluate.py`: Evaluation logic and metrics
- `utils.py`: Utility functions
- `config.py`: Configuration management
- `requirements.txt`: Python package dependencies

Please generate the complete codebase as specified. After generation, write a confirmation message "AGENTIC_CODING_COMPLETE" and then exit.
"""
        return prompt
    
    def generate_code_with_cli(self, prompt: str) -> bool:
        """
        Generates code by running the agentic coder CLI in a Docker container
        and feeding it the prompt via stdin.
        """
        if not self.docker_client:
            return False

        # Prepare environment variables for the Docker container
        env_vars = {
            "SANDBOX_RUNTIME_CONTAINER_IMAGE": "docker.all-hands.dev/all-hands-ai/runtime:0.45-nikolaik",
            "SANDBOX_USER_ID": f"{os.getuid()}" if hasattr(os, 'getuid') else "1000",
            "LLM_MODEL": os.getenv("MODEL_NAME", "anthropic/claude-3-sonnet-20240229"),
            "LLM_API_KEY": os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        }
        if not env_vars["LLM_API_KEY"]:
            logger.error("No LLM_API_KEY found in environment variables. Cannot proceed.")
            return False

        # Docker command arguments
        docker_command = [
            "docker", "run", "-i", "--rm",
            "--pull=always",
            "--name", self.container_name,
            "-e", f"SANDBOX_RUNTIME_CONTAINER_IMAGE={env_vars['SANDBOX_RUNTIME_CONTAINER_IMAGE']}",
            "-e", f"SANDBOX_USER_ID={env_vars['SANDBOX_USER_ID']}",
            "-e", f"LLM_MODEL={env_vars['LLM_MODEL']}",
            "-e", f"LLM_API_KEY={env_vars['LLM_API_KEY']}",
            "-v", "/var/run/docker.sock:/var/run/docker.sock",
            "-v", f"{Path.home()}/.openhands:/.openhands",
            "-v", f"{self.output_dir}:/app/workspace",
            "--add-host", "host.docker.internal:host-gateway",
            "docker.all-hands.dev/all-hands-ai/openhands:0.45",
            "python", "-m", "openhands.cli.main", "--workspace-dir", "/app/workspace"
        ]
        
        logger.info("Starting agentic coder CLI in Docker...")
        logger.info(f"Output will be generated in: {self.output_dir}")
        
        try:
            # Run the Docker command as a subprocess
            process = subprocess.Popen(
                docker_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # Feed the prompt to the CLI via stdin
            logger.info("Sending prompt to the agent...")
            
            # The CLI might need some initial commands before the main prompt
            initial_commands = "/init\n" # Initialize the repository context
            
            stdout, stderr = process.communicate(input=initial_commands + prompt + "\n/exit\n", timeout=3600) # 1 hour timeout

            logger.info("Agent process finished.")
            logger.debug(f"Agent STDOUT:\n{stdout}")
            
            if process.returncode != 0:
                logger.error("Agentic coder process exited with a non-zero status code.")
                logger.error(f"Agent STDERR:\n{stderr}")
                return False

            if "AGENTIC_CODING_COMPLETE" not in stdout:
                 logger.warning("Confirmation message not found in agent's output. Code generation may be incomplete.")
                 # We can still return True if files were generated.
            
            # Verify that files were created
            if not any(self.output_dir.iterdir()):
                logger.error("The output directory is empty. Code generation failed.")
                return False

            logger.info("Code generation appears to be successful.")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Code generation timed out after 1 hour.")
            process.kill()
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while running the agentic coder: {e}", exc_info=True)
            return False

    def generate_code_from_plan(self, plan_file: str) -> bool:
        """
        Main automated workflow to generate code from a plan using the CLI.
        """
        logger.info("Starting automated CLI-based code generation...")
        
        try:
            with open(plan_file, 'r', encoding='utf-8') as f:
                plan_data = json.load(f)
            
            prompt = self.convert_plan_to_prompt(plan_data)
            
            success = self.generate_code_with_cli(prompt)
            
            return success

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            return False

def main():
    """
    Main function for testing the agentic coder integration.
    """
    logger.info("--- Agentic Coder CLI Integration Test ---")
    
    # This path needs to be valid for the test to run.
    plan_file = "BioForge/Method_Design/example_report.json"
    output_dir = "generated_code_cli_test"
    
    if not Path(plan_file).exists():
        logger.error(f"Test plan not found at '{plan_file}'.")
        logger.error("Please provide a valid path to a method design plan JSON file to run the test.")
        return
        
    logger.info(f"Test output will be saved to: {output_dir}")
    
    # Pass the output directory during initialization
    coder = AgenticCoder(output_dir=output_dir)
    success = coder.generate_code_from_plan(plan_file)
    
    if success:
        logger.info("--- Test Completed Successfully ---")
    else:
        logger.error("--- Test Failed ---")

if __name__ == "__main__":
    main()