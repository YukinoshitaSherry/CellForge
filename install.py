#!/usr/bin/env python3
"""
cellforge Installation Script
Simplified installation and configuration process
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# Try to load dotenv, install if not available
try:
    from dotenv import load_dotenv
except ImportError:
    print("Installing python-dotenv...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
try:
    load_dotenv()
except Exception as e:
    print(f"⚠️  Warning: Could not load .env file: {e}")
    print("   This is normal if .env file doesn't exist yet")

# Default task description - EDIT THIS VARIABLE TO CUSTOMIZE YOUR TASK
DEFAULT_TASK_DESCRIPTION = """Your task is to develop a predictive model that accurately estimates gene expression profiles of individual K562 cells following CRISPR interference (CRISPRi), using the dataset from Norman et al. (2019, Science).

Task Definition:
- Input: Baseline gene expression profile of an unperturbed K562 cell and the identity of the target gene(s) for perturbation
- Output: Predicted gene expression profile after perturbation

Evaluation Scenarios:
1. Unseen Perturbations: Predict effects of gene perturbations not present during training
2. Unseen Cell Contexts: Predict responses in cells with gene expression profiles not observed during training

Evaluation Metrics:
- Mean Squared Error (MSE): Measures the average squared difference between predicted and observed gene expression.
- Pearson Correlation Coefficient (PCC): Quantifies linear correlation between predicted and observed profiles.
- R² (Coefficient of Determination): Represents the proportion of variance in the observed gene expression that can be explained by the predicted values.
- MSE for Differentially Expressed (DE) Genes (MSE_DE): Same as MSE but computed specifically for genes identified as differentially expressed.
- PCC for Differentially Expressed (DE) Genes (PCC_DE): Same as PCC but computed specifically for genes identified as differentially expressed.
- R² for Differentially Expressed (DE) Genes (R2_DE): Same as R² but computed specifically for genes identified as differentially expressed."""

def run_command(command, description):
    """Run command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

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
        
        response = input("Continue without virtual environment? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            print("⚠️  Proceeding without virtual environment")
            return True
        else:
            print("❌ Please set up a virtual environment first")
            return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directory structure...")
    
    directories = [
        "cellforge/data/datasets/scRNA-seq",
        "cellforge/data/datasets/scATAC-seq", 
        "cellforge/data/datasets/perturbation",
        "cellforge/data/papers/pdf",
        "cellforge/data/code",
        "cellforge/data/results",
        "results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")

def setup_environment():
    """Set up environment variables"""
    print("⚙️ Setting up environment...")
    
    # Check .env file
    if not Path(".env").exists():
        if Path("env.example").exists():
            shutil.copy("env.example", ".env")
            print("✅ Created .env file from template")
            print("⚠️  IMPORTANT: Please edit .env file to add your API keys")
            print("   📍 Location: .env (project root directory)")
            print("   📝 Edit with: nano .env, code .env, or notepad .env")
            print("   🔑 Required: At least one LLM API key + SerpAPI key")
            print("   🔑 Optional: GitHub token and PubMed API key")
        else:
            print("⚠️  No env.example found, please create .env file manually")
            print("   📍 Location: .env (project root directory)")
    else:
        print("✅ .env file already exists")
        print("   📍 Location: .env (project root directory)")
        print("   🔍 Please verify your API keys are configured correctly")

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # First install core dependencies using python -m pip to avoid permission issues
    if not run_command(f"{sys.executable} -m pip install --upgrade pip setuptools wheel", "Upgrading pip, setuptools, and wheel"):
        return False
    
    # Install requirements.txt
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    # Install cellforge in normal mode (not development mode)
    if not run_command("pip install .", "Installing cellforge"):
        return False
    
    return True

def create_config_file():
    """Create configuration file"""
    print("📝 Creating configuration file...")
    
    config_content = {
        "task_description": DEFAULT_TASK_DESCRIPTION,
        "dataset_path": "cellforge/data/datasets/",
        "output_dir": "results/",
        "llm_config": {
            "provider": "openai",
            "model": os.getenv("MODEL_NAME", "gpt-4"),
            "api_key": "loaded_from_env"  # API keys are loaded from .env file
        },
        "workflow_phases": ["task_analysis", "method_design", "code_generation"],
        "qdrant_config": {
            "host": os.getenv("QDRANT_URL", "localhost"),
            "port": int(os.getenv("QDRANT_PORT", "6333"))
        }
    }
    
    import json
    with open("config.json", 'w', encoding='utf-8') as f:
        json.dump(config_content, f, ensure_ascii=False, indent=2)
    
    print("✅ Configuration file config.json created")
    print("⚠️  Please configure your API keys in .env file")
    print("💡 To customize your task, edit the DEFAULT_TASK_DESCRIPTION variable in main.py")

def verify_installation():
    """Verify installation"""
    print("🔍 Verifying installation...")
    
    # Test imports
    try:
        import cellforge
        print("✅ cellforge package imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import cellforge: {e}")
        return False
    
    # Test basic functionality
    try:
        from cellforge.Task_Analysis.main import run_task_analysis
        print("✅ cellforge functions imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import cellforge functions: {e}")
        return False
    
    return True

def main():
    """Main installation function"""
    print("🚀 cellforge Installation Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check virtual environment
    if not check_virtual_environment():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Set up environment
    setup_environment()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create configuration file
    create_config_file()
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        sys.exit(1)
    
    print("\n🎉 cellforge installation completed!")
    print("\n📋 Next steps:")
    print("1. 📝 Edit .env file to set your API keys")
    print("   📍 Location: .env (project root directory)")
    print("   🔑 Required: At least one LLM API key + SerpAPI key")
    print("2. ✏️  To customize your task, edit the DEFAULT_TASK_DESCRIPTION variable in main.py")
    print("3. 📁 Place your datasets in cellforge/data/datasets/ directory")
    print("4. 🚀 Run: python main.py")
    print("5. ✅ Verify setup: python start.py")
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main() 