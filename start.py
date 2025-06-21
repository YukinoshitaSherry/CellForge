#!/usr/bin/env python3
"""
BioForge Quick Start Script
Simple script to test if BioForge is properly installed and configured
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_virtual_environment():
    """Check if running in a virtual environment"""
    print("🔍 Checking virtual environment...")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if in_venv:
        print("✅ Running in virtual environment")
        print(f"   Virtual environment: {sys.prefix}")
        return True
    else:
        print("⚠️  Not running in virtual environment")
        print("💡 It's recommended to use a virtual environment to avoid dependency conflicts")
        print("   You can create one with:")
        print("   - conda: conda create -n bioforge python=3.9")
        print("   - venv: python -m venv venv")
        print("   - pipenv: pipenv install")
        return False

def check_installation():
    """Check if BioForge is properly installed"""
    print("🔍 Checking BioForge installation...")
    
    # Check if main modules can be imported
    try:
        import BioForge
        print("✅ BioForge package imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import BioForge: {e}")
        return False
    
    # Check if Task Analysis module can be imported
    try:
        from BioForge.Task_Analysis.main import run_task_analysis
        print("✅ Task Analysis module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Task Analysis module: {e}")
        return False
    
    return True

def check_config():
    """Check if configuration file exists and is valid"""
    print("\n📝 Checking configuration...")
    
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        print("💡 Run 'python main.py --init' to create default configuration")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print("✅ Configuration file found and valid")
        
        # Check required fields
        required_fields = ["task_description", "dataset_path", "llm_config"]
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                return False
        
        print("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return False

def check_env_config():
    """Check if .env file exists and API keys are configured"""
    print("\n🔑 Checking API configuration...")
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("❌ .env file not found")
        print("💡 Please copy env.example to .env and configure your API keys")
        print("   📍 Location: .env (project root directory)")
        print("   📝 Command: cp env.example .env")
        return False
    
    print("✅ .env file found")
    print("   📍 Location: .env (project root directory)")
    
    # Check if at least one LLM API key is configured
    llm_api_keys = [
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("DEEPSEEK_API_KEY"),
        os.getenv("LLAMA_API_KEY"),
        os.getenv("QWEN_API_KEY")
    ]
    
    configured_llm_keys = [key for key in llm_api_keys if key and key != "your_openai_api_key_here"]
    
    if not configured_llm_keys:
        print("⚠️  No LLM API keys configured in .env file")
        print("💡 Please edit .env file and add at least one LLM API key")
        print("   🔑 Required: At least one LLM API key (OpenAI, Anthropic, etc.)")
        return False
    
    print(f"✅ {len(configured_llm_keys)} LLM API key(s) configured")
    
    # Check if all required search API keys are configured
    search_api_keys = {
        "GitHub": os.getenv("GITHUB_TOKEN"),
        "SerpAPI": os.getenv("SERPAPI_KEY"),
        "PubMed": os.getenv("PUBMED_API_KEY")
    }
    
    missing_search_keys = []
    for name, key in search_api_keys.items():
        if not key or key == f"your_{name.lower()}_key_here":
            if name == "SerpAPI":  # SerpAPI is required
                missing_search_keys.append(name)
            elif name in ["GitHub", "PubMed"]:  # GitHub and PubMed are optional
                print(f"⚠️  {name} API key not configured (optional)")
    
    if missing_search_keys:
        print(f"⚠️  Missing required search API keys: {', '.join(missing_search_keys)}")
        print("💡 SerpAPI key is required for RAG functionality")
        return False
    
    print("✅ All required search API keys configured")
    return True

def check_directories():
    """Check if required directories exist"""
    print("\n📁 Checking directory structure...")
    
    required_dirs = [
        "BioForge/data/datasets/scRNA-seq",
        "BioForge/data/datasets/scATAC-seq",
        "BioForge/data/datasets/perturbation",
        "results"
    ]
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}")
        else:
            print(f"❌ {directory} - missing")
            return False
    
    return True

def run_test():
    """Run a simple test to verify functionality"""
    print("\n🧪 Running functionality test...")
    
    try:
        # Import the main function
        from main import load_config, validate_config
        
        # Load configuration
        config = load_config()
        
        # Validate configuration
        if validate_config(config):
            print("✅ Configuration validation passed")
        else:
            print("❌ Configuration validation failed")
            return False
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 BioForge Quick Start Check")
    print("=" * 50)
    
    # Check virtual environment
    if not check_virtual_environment():
        print("\n❌ Virtual environment check failed")
        return
    
    # Check installation
    if not check_installation():
        print("\n❌ Installation check failed")
        print("💡 Run 'python install.py' to install BioForge")
        return
    
    # Check directories
    if not check_directories():
        print("\n❌ Directory check failed")
        print("💡 Run 'python main.py --init' to create directory structure")
        return
    
    # Check environment configuration
    if not check_env_config():
        print("\n❌ Environment configuration check failed")
        print("💡 Please configure your API keys in .env file")
        return
    
    # Check configuration
    if not check_config():
        print("\n❌ Configuration check failed")
        print("💡 Run 'python main.py --init' to create configuration")
        return
    
    # Run test
    if not run_test():
        print("\n❌ Functionality test failed")
        return
    
    print("\n" + "=" * 50)
    print("🎉 All checks passed! BioForge is ready to use.")
    print("=" * 50)
    print("\n📋 Next steps:")
    print("1. Place your datasets in BioForge/data/datasets/")
    print("2. Run: python main.py")
    print("3. Check results in the results/ directory")

if __name__ == "__main__":
    main() 