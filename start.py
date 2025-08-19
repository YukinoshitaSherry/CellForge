#!/usr/bin/env python3
"""
cellforge Quick Start Script
Simple script to test if cellforge is properly installed and configured
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
try:
    load_dotenv()
except Exception as e:
    print(f"⚠️  Warning: Could not load .env file: {e}")
    print("   This is normal if .env file doesn't exist yet")

def check_virtual_environment():
    """Check if running in a virtual environment"""
    print("🔍 Checking virtual environment...")
    
    # Check if we're in a virtual environment (improved detection for Windows)
    in_venv = False
    
    # Method 1: Check for real_prefix (venv)
    if hasattr(sys, 'real_prefix'):
        in_venv = True
    # Method 2: Check for base_prefix != prefix (conda and some venv)
    elif hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        in_venv = True
    # Method 3: Check for VIRTUAL_ENV environment variable
    elif os.getenv('VIRTUAL_ENV'):
        in_venv = True
    # Method 4: Check for CONDA_DEFAULT_ENV environment variable
    elif os.getenv('CONDA_DEFAULT_ENV'):
        in_venv = True
    # Method 5: Check if Python executable is in a venv-like directory structure
    else:
        python_path = sys.executable
        if 'env' in python_path.lower() or 'venv' in python_path.lower() or 'conda' in python_path.lower():
            in_venv = True
    
    if in_venv:
        print("✅ Running in virtual environment")
        if os.getenv('CONDA_DEFAULT_ENV'):
            print(f"   Conda environment: {os.getenv('CONDA_DEFAULT_ENV')}")
        elif os.getenv('VIRTUAL_ENV'):
            print(f"   Virtual environment: {os.getenv('VIRTUAL_ENV')}")
        else:
            print(f"   Python executable: {sys.executable}")
        return True
    else:
        print("⚠️  Not running in virtual environment")
        print("💡 It's recommended to use a virtual environment to avoid dependency conflicts")
        print("   You can create one with:")
        print("   - conda: conda create -n cellforge python=3.9")
        print("   - venv: python -m venv venv")
        print("   - pipenv: pipenv install")
        return False

def check_installation():
    """Check if cellforge is properly installed"""
    print("🔍 Checking cellforge installation...")
    
    # Check if main modules can be imported
    try:
        import cellforge
        print("✅ cellforge package imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import cellforge: {e}")
        return False
    
    # Check if Task Analysis module can be imported
    try:
        from cellforge.Task_Analysis.main import run_task_analysis
        print("✅ Task Analysis module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Task Analysis module: {e}")
        # Try alternative import
        try:
            import cellforge.Task_Analysis
            print("✅ Task Analysis module imported successfully (alternative method)")
        except ImportError as e2:
            print(f"❌ Failed to import Task Analysis module: {e2}")
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
    
    # Check search API keys (optional for basic functionality)
    search_api_keys = {
        "GitHub": os.getenv("GITHUB_TOKEN"),
        "SerpAPI": os.getenv("SERPAPI_KEY"),
        "PubMed": os.getenv("PUBMED_API_KEY")
    }
    
    missing_search_keys = []
    configured_search_keys = []
    for name, key in search_api_keys.items():
        if not key or key == f"your_{name.lower()}_key_here":
            missing_search_keys.append(name)
        else:
            configured_search_keys.append(name)
    
    if missing_search_keys:
        print(f"⚠️  Missing optional search API keys: {', '.join(missing_search_keys)}")
        print("💡 These keys are optional for enhanced RAG functionality")
        if configured_search_keys:
            print(f"✅ Configured search APIs: {', '.join(configured_search_keys)}")
        else:
            print("⚠️  No search APIs configured - RAG functionality will be limited")
    
    return True

def check_directories():
    """Check if required directories exist"""
    print("\n📁 Checking directory structure...")
    
    required_dirs = [
        "cellforge/data/datasets/scRNA-seq",
        "cellforge/data/datasets/scATAC-seq",
        "cellforge/data/datasets/perturbation",
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
        # Test basic import functionality
        from cellforge.Task_Analysis.main import run_task_analysis
        print("✅ Task Analysis module imported successfully")
        
        # Test configuration loading
        config_path = "config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print("✅ Configuration loading test passed")
        else:
            print("⚠️  Configuration file not found, but this is optional")
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 cellforge Quick Start Check")
    print("=" * 50)
    
    # Check virtual environment
    if not check_virtual_environment():
        print("\n❌ Virtual environment check failed")
        return
    
    # Check installation
    if not check_installation():
        print("\n❌ Installation check failed")
        print("💡 Run 'python install.py' to install cellforge")
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
    print("🎉 All checks passed! cellforge is ready to use.")
    print("=" * 50)
    print("\n📋 Next steps:")
    print("1. Place your datasets in cellforge/data/datasets/")
    print("2. Run: python main.py")
    print("3. Check results in the results/ directory")

if __name__ == "__main__":
    main() 