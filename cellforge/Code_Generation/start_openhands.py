#!/usr/bin/env python3
"""
OpenHands Startup Script for CellForge Code Generation
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_docker():
    """Check if Docker is available"""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            logger.info(f"✅ Docker available: {result.stdout.strip()}")
            return True
        else:
            logger.error("❌ Docker not available")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.error("❌ Docker not found or not accessible")
        return False

def check_docker_compose():
    """Check if docker-compose is available"""
    try:
        result = subprocess.run(
            ["docker-compose", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            logger.info(f"✅ Docker Compose available: {result.stdout.strip()}")
            return True
        else:
            logger.error("❌ Docker Compose not available")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.error("❌ Docker Compose not found or not accessible")
        return False

def start_openhands():
    """Start OpenHands Docker container"""
    try:
        # Get OpenHands directory
        openhands_path = Path(__file__).parent / "OpenHands"
        
        if not openhands_path.exists():
            logger.error(f"❌ OpenHands directory not found: {openhands_path}")
            return False
        
        docker_compose_file = openhands_path / "docker-compose.yml"
        if not docker_compose_file.exists():
            logger.error(f"❌ docker-compose.yml not found in: {openhands_path}")
            return False
        
        logger.info(f"📁 OpenHands directory: {openhands_path}")
        
        # Change to OpenHands directory
        original_cwd = Path.cwd()
        os.chdir(openhands_path)
        
        try:
            # Stop any existing containers first
            logger.info("🛑 Stopping any existing OpenHands containers...")
            subprocess.run(
                ["docker-compose", "down"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Start Docker container using docker-compose
            logger.info("🚀 Building and starting OpenHands container...")
            cmd = ["docker-compose", "up", "-d", "--build"]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout for build
            )
            
            if result.returncode == 0:
                logger.info("✅ OpenHands Docker container started successfully")
                if result.stdout:
                    logger.info(f"📋 Container output: {result.stdout}")
                return True
            else:
                logger.error(f"❌ Failed to start OpenHands: {result.stderr}")
                if result.stdout:
                    logger.error(f"📋 Command output: {result.stdout}")
                return False
                
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout starting OpenHands Docker container")
        return False
    except Exception as e:
        logger.error(f"❌ Error starting OpenHands: {e}")
        return False

def wait_for_openhands_ready(timeout: int = 180):
    """Wait for OpenHands to be ready"""
    logger.info("⏳ Waiting for OpenHands to be ready...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Check if OpenHands is responding
            response = requests.get("http://localhost:3000", timeout=10)
            if response.status_code == 200:
                logger.info("✅ OpenHands is ready!")
                return True
            else:
                logger.info(f"📡 OpenHands responding with status: {response.status_code}")
        except requests.RequestException as e:
            logger.info(f"⏳ OpenHands not ready yet: {e}")
        
        time.sleep(5)
    
    logger.error("❌ OpenHands did not become ready within timeout")
    return False

def test_openhands_api():
    """Test OpenHands API"""
    try:
        logger.info("🧪 Testing OpenHands API...")
        
        # Test basic connectivity
        response = requests.get("http://localhost:3000", timeout=10)
        if response.status_code == 200:
            logger.info("✅ OpenHands API is responding")
            
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
                timeout=60
            )
            
            if response.status_code == 200:
                logger.info("✅ OpenHands chat API is working")
                return True
            else:
                logger.error(f"❌ OpenHands chat API error: {response.status_code} - {response.text}")
                return False
        else:
            logger.error(f"❌ OpenHands API not responding properly: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing OpenHands API: {e}")
        return False

def main():
    """Main function"""
    logger.info("=== OpenHands Startup Script ===")
    
    # Check prerequisites
    logger.info("🔍 Checking prerequisites...")
    if not check_docker():
        logger.error("❌ Docker is required but not available")
        return False
    
    if not check_docker_compose():
        logger.error("❌ Docker Compose is required but not available")
        return False
    
    # Start OpenHands
    logger.info("🚀 Starting OpenHands...")
    if not start_openhands():
        logger.error("❌ Failed to start OpenHands")
        return False
    
    # Wait for OpenHands to be ready
    if not wait_for_openhands_ready():
        logger.error("❌ OpenHands did not become ready")
        return False
    
    # Test OpenHands API
    if not test_openhands_api():
        logger.error("❌ OpenHands API test failed")
        return False
    
    logger.info("🎉 OpenHands is ready for use!")
    logger.info("🌐 OpenHands is available at: http://localhost:3000")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ OpenHands startup completed successfully!")
        print("You can now use OpenHands for code generation.")
    else:
        print("\n❌ OpenHands startup failed!")
        print("Please check the logs above for details.")
        sys.exit(1)
