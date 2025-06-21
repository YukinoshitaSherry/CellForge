![BioForge Workflow](./Figs/bioforge.png){ width=10% }
# BioForge: Open-Ended Autonomous Design of Computational Methods for Single-Cell Omics via Multi-Agent Collaboration

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

BioForge is a cutting-edge **end-to-end** multi-agent framework that revolutionizes single-cell data analysis through intelligent task decomposition, automated method design, and collaborative problem-solving. The system integrates advanced language models, vector databases, and domain-specific expertise to provide comprehensive solutions for single-cell genomics challenges.

![BioForge Workflow](./Figs/Bioforge_workflow.png){ width=90% }

## 🚀 Installation

### ✨ Virtual Environment Setup

Before installation, create and activate a virtual environment:

```bash
# Create virtual environment
conda create -n bioforge python=3.9
conda activate bioforge

# Or using venv (Python built-in)
python -m venv bioforge_env
# On Windows:
bioforge_env\Scripts\activate
# On macOS/Linux:
source bioforge_env/bin/activate
```

### 📦 Quick Installation (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/gersteinlab/scAgents.git
cd BioForge

# 2. Create and activate virtual environment
conda create -n bioforge python=3.9
conda activate bioforge

# 3. Run automated installation
python install.py

# 4. Configure API keys
cp env.example .env
# Edit .env with your API keys

# 5. Verify installation
python start.py
```

### ⚙️ Manual Installation (Alternative)

If automated installation fails, try manual installation:

```bash
# 1. Install minimal dependencies
pip install -r requirements-minimal.txt

# 2. Install BioForge
pip install -e .

# 3. Configure environment
cp env.example .env
# Edit .env with your API keys
```

### 🔑 Prerequisites

- **Python**: 3.8 or higher (3.9 recommended)
- **Memory**: 8GB RAM minimum (16GB+ recommended)
- **Storage**: 10GB free space
- **API Keys**: OpenAI, Anthropic, or other LLM service API key
- **Docker**: For code generation
  
## ⚡ Quick Start

### 1️⃣ Verify Installation

Run the quick start check to ensure everything is properly set up:

```bash
python start.py
```

This will check:
- ✅ BioForge installation
- ✅ Directory structure
- ✅ Environment configuration (.env file)
- ✅ Configuration file
- ✅ Basic functionality
- ✅ Virtual environment status

### 2️⃣ Configure API Keys

**Step 1: Copy the environment template to project root directory:**

```bash
# Make sure you're in the BioForge project root directory
cd /path/to/your/BioForge
cp .env.example .env
```


**Step 2: Add your API keys to the `.env` file:**

```bash
# Example .env file content
# LLM API Keys (at least one required)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key
LLAMA_API_KEY=your_llama_api_key
QWEN_API_KEY=your_qwen_api_key

# Search API Keys (required for RAG functionality)
GITHUB_TOKEN=your_github_token_here
SERPAPI_KEY=your_serpapi_key_here
PUBMED_API_KEY=your_pubmed_api_key_here

# Model Configuration
MODEL_NAME=gpt-4-turbo-preview
TEMPERATURE=0.7
MAX_TOKENS=4096000
```

**API Key Requirements:**

**LLM API Keys (choose at least one):**
- OpenAI 
- Anthropic (Claude)

**Search API Keys:**
- **GitHub Token**: For searching GitHub repositories and code (optional)
- **SerpAPI Key**: For web search functionality (required)

**How to get API keys:**
- **OpenAI**: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Anthropic**: [https://console.anthropic.com/](https://console.anthropic.com/)
- **GitHub**: [https://github.com/settings/tokens](https://github.com/settings/tokens)
- **SerpAPI**: [https://serpapi.com/](https://serpapi.com/)

### 3️⃣ Run BioForge

**Complete Workflow:**
```bash
python main.py
```

**Individual Phases:**
```bash
# Task Analysis only
python main.py --phase task_analysis

# Method Design only
python main.py --phase method_design

# Code Generation only
python main.py --phase code_generation
```

**Initialize Project:**
```bash
python main.py --init
```

## 🔧 Usage

### Command Line Interface

**Complete Workflow**:
```bash
python main.py
```

**With Custom Config**:
```bash
python main.py --config my_config.json
```

**Initialize Project**:
```bash
python main.py --init
```

**Verify Setup**:
```bash
python start.py
```


## 🔍 Troubleshooting

### Common Issues

**1. Virtual Environment Issues**
```bash
# Check if virtual environment is activated
which python  # Should show path to your venv
echo $VIRTUAL_ENV  # Should show venv path

# If not activated, activate it:
# For conda:
conda activate bioforge
# For venv:
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

**2. Import Errors**
```bash
# Ensure you're in the correct directory and venv is activated
cd BioForge
conda activate bioforge  # or source venv/bin/activate
pip install -e .
```

**3. API Key Issues**
```bash
# Check your .env file
cat .env | grep API_KEY
```

**4. Missing Dependencies**
```bash
# Reinstall dependencies in your virtual environment
conda activate bioforge  # or source venv/bin/activate
pip install -r requirements.txt
```

**5. Dataset Path Issues**
```bash
# Check dataset directory structure
ls -la BioForge/data/datasets/
```

**6. Environment Configuration**
```bash
# Verify .env file exists and is configured
python start.py

# Test PubMed API specifically
python test_pubmed.py
```

**7. Qdrant Connection Issues**
```bash
# Start Qdrant if not running
docker run -p 6333:6333 qdrant/qdrant

# Check if Qdrant is accessible
curl http://localhost:6333/collections
```

**8. Docker Issues**
```bash
# Check if Docker is running
docker --version
docker ps

# If Docker is not running, start it:
# On Windows/Mac: Start Docker Desktop
# On Linux: sudo systemctl start docker
```

## 📚 Citation

If you use BioForge in your research, please cite:

```bibtex
@article{bioForge2024,
    title = {BioForge: Open-Ended Autonomous Design of Computational Methods for Single-Cell Omics via Multi-Agent Collaboration},
    author = {Tang, Xiangru and Yu, Zhuoyun and Chen, Jiapeng and Cui, Yan and Shao, Daniel and Wu, Fang and Wang, Weixu and Huang, Zhi and Cohan, Arman and Krishnaswamy, Smita and Gerstein, Mark},
    year = {2024},
    journal = {arXiv preprint}
}
```

**For datasets from scPerturb:**
```bibtex
@article{Peidli2024scperturb,
    title={scPerturb: harmonized single-cell perturbation data},
    author={Peidli, Stefan and Green, Tessa D. and Shen, Ciyue and Gross, Torsten and Min, Joseph and others},
    journal={Nature Methods},
    volume={21},
    number={3},
    pages={531--540},
    year={2024},
    doi={10.1038/s41592-023-02144-y},
    url={https://www.nature.com/articles/s41592-023-02144-y}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
