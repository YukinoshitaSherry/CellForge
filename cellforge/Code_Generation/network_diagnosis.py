#!/usr/bin/env python3
"""
Network Diagnosis Script for OpenHands
Helps diagnose and resolve network connectivity issues
"""

import subprocess
import requests
import urllib.request
import socket
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NetworkDiagnosis:
    """Network diagnosis and troubleshooting for OpenHands"""
    
    def __init__(self):
        self.issues = []
        self.solutions = []
    
    def run_full_diagnosis(self):
        """Run complete network diagnosis"""
        logger.info("=== OpenHands Network Diagnosis ===")
        
        # Basic connectivity tests
        self.test_basic_connectivity()
        self.test_dns_resolution()
        self.test_docker_registries()
        self.test_proxy_settings()
        self.test_firewall_settings()
        
        # Report results
        self.generate_report()
    
    def test_basic_connectivity(self):
        """Test basic internet connectivity"""
        logger.info("\n--- Basic Connectivity Tests ---")
        
        # Test Google
        try:
            urllib.request.urlopen('https://www.google.com', timeout=10)
            logger.info("✓ Google connectivity: OK")
        except Exception as e:
            logger.error(f"✗ Google connectivity failed: {e}")
            self.issues.append("Basic internet connectivity failed")
            self.solutions.append("Check your internet connection and try again")
        
        # Test GitHub
        try:
            urllib.request.urlopen('https://github.com', timeout=10)
            logger.info("✓ GitHub connectivity: OK")
        except Exception as e:
            logger.error(f"✗ GitHub connectivity failed: {e}")
            self.issues.append("GitHub connectivity failed")
            self.solutions.append("Check if GitHub is accessible from your network")
    
    def test_dns_resolution(self):
        """Test DNS resolution"""
        logger.info("\n--- DNS Resolution Tests ---")
        
        domains = [
            'ghcr.io',
            'registry-1.docker.io',
            'docker.all-hands.dev',
            'github.com'
        ]
        
        for domain in domains:
            try:
                ip = socket.gethostbyname(domain)
                logger.info(f"✓ DNS resolution for {domain}: {ip}")
            except socket.gaierror as e:
                logger.error(f"✗ DNS resolution failed for {domain}: {e}")
                self.issues.append(f"DNS resolution failed for {domain}")
                self.solutions.append("Try using a different DNS server (e.g., 8.8.8.8 or 1.1.1.1)")
    
    def test_docker_registries(self):
        """Test Docker registry connectivity"""
        logger.info("\n--- Docker Registry Tests ---")
        
        registries = [
            ('https://ghcr.io', 'GitHub Container Registry'),
            ('https://registry-1.docker.io', 'Docker Hub'),
            ('https://docker.all-hands.dev', 'OpenHands Registry')
        ]
        
        for url, name in registries:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code < 400:
                    logger.info(f"✓ {name} connectivity: OK")
                else:
                    logger.warning(f"⚠ {name} returned status: {response.status_code}")
            except Exception as e:
                logger.error(f"✗ {name} connectivity failed: {e}")
                self.issues.append(f"{name} connectivity failed")
                self.solutions.append(f"Check if {name} is accessible from your network")
    
    def test_proxy_settings(self):
        """Check proxy settings"""
        logger.info("\n--- Proxy Settings Check ---")
        
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
        proxy_found = False
        
        for var in proxy_vars:
            value = os.environ.get(var)
            if value:
                logger.info(f"Found proxy setting: {var}={value}")
                proxy_found = True
        
        if not proxy_found:
            logger.info("✓ No proxy settings detected")
        else:
            logger.warning("⚠ Proxy settings detected - this may affect Docker image pulling")
            self.solutions.append("If you're behind a corporate proxy, configure Docker to use it")
    
    def test_firewall_settings(self):
        """Check firewall and security software"""
        logger.info("\n--- Firewall/Security Check ---")
        
        # Test if Docker can access external networks
        try:
            result = subprocess.run(
                ["docker", "run", "--rm", "alpine:latest", "ping", "-c", "1", "8.8.8.8"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                logger.info("✓ Docker network access: OK")
            else:
                logger.error("✗ Docker network access failed")
                self.issues.append("Docker cannot access external networks")
                self.solutions.append("Check firewall settings and allow Docker through")
        except Exception as e:
            logger.error(f"✗ Docker network test failed: {e}")
            self.issues.append("Docker network test failed")
            self.solutions.append("Ensure Docker is running and has network access")
    
    def generate_report(self):
        """Generate diagnosis report"""
        logger.info("\n=== Diagnosis Report ===")
        
        if not self.issues:
            logger.info("✓ No network issues detected!")
            logger.info("If OpenHands still fails to start, try:")
            logger.info("1. Restart Docker Desktop")
            logger.info("2. Clear Docker cache: docker system prune -a")
            logger.info("3. Try again in a few minutes")
        else:
            logger.error(f"Found {len(self.issues)} issue(s):")
            for i, issue in enumerate(self.issues, 1):
                logger.error(f"{i}. {issue}")
            
            logger.info("\nSuggested solutions:")
            for i, solution in enumerate(self.solutions, 1):
                logger.info(f"{i}. {solution}")
            
            logger.info("\nAdditional troubleshooting steps:")
            logger.info("1. Try using a VPN if you're in a restricted network")
            logger.info("2. Check if your antivirus is blocking Docker")
            logger.info("3. Try running as administrator")
            logger.info("4. Restart your computer and try again")

def main():
    """Main function"""
    import os
    
    diagnosis = NetworkDiagnosis()
    diagnosis.run_full_diagnosis()

if __name__ == "__main__":
    main()
