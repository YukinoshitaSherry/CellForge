from setuptools import setup, find_packages

setup(
    name="BioForge",
    version="0.1.0",
    packages=find_packages(include=['BioForge', 'BioForge.*']),
    install_requires=[
        # Core Dependencies
        "setuptools>=65.5.1",
        "wheel>=0.38.4",
        "packaging>=23.0",
        "filelock>=3.9.0",
        "typing_extensions>=4.5.0",
        "colorama>=0.4.6",
        "charset-normalizer>=2.0.0",
        
        # HTTP & Async Support
        "urllib3>=1.26.18,<2.0.0",
        "requests>=2.31.0",
        "aiohttp>=3.8.0",
        "aiosignal>=1.1.2",
        "async-timeout>=4.0.2",
        "frozenlist>=1.3.3",
        "multidict>=6.0.4",
        "yarl>=1.8.2",
        "aiofiles>=23.1.0",
        "attrs>=23.1.0",
        "tenacity>=8.2.0,!=8.4.0,<10.0.0",
        "idna>=3.6",
        "certifi>=2023.7.22",
        
        # Configuration & File Handling
        "python-dotenv>=0.19.0",
        "pyyaml>=6.0.1",
        "json5>=0.9.14",
        "jsonschema>=4.22.0",
        "jsonschema-specifications>=2023.7.1",
        "referencing>=0.30.2",
        "rpds-py>=0.10.3",
        
        # Scientific Computing
        "numpy>=1.24.3,<2.0.0",
        "scipy>=1.10.1",
        "pandas>=2.2.3",
        "scikit-learn>=1.3.0",
        "python-dateutil>=2.8.2",
        "pytz>=2023.3",
        "tzdata>=2023.3",
        "h5py>=3.8.0",
        "joblib>=1.2.0",
        "threadpoolctl>=3.1.0",
        "networkx>=2.8.8",
        "numba>=0.56.4",
        "llvmlite>=0.39.1",
        
        # Deep Learning & NLP
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "torchaudio>=2.0.2",
        "transformers>=4.30.0",
        "tokenizers>=0.13.3",
        "sentence-transformers>=2.2.2",
        "safetensors>=0.3.1",
        "huggingface-hub>=0.16.4",
        "nltk>=3.8.1",
        "regex>=2023.3.23",
        "sentencepiece>=0.1.97",
        
        # Vector Store & Search
        "qdrant-client>=1.7.0",
        "PyPDF2>=3.0.0",
        "pdfplumber>=0.10.0",
        
        # Single Cell Analysis
        "scanpy>=1.9.3",
        "anndata>=0.9.2",
        "scib>=1.1.7",
        "gseapy>=1.1.0",
        "scvi-tools>=1.0.0",
        "patsy>=0.5.3",
        "session-info>=1.0.0",
        "statsmodels>=0.13.5",
        "umap-learn>=0.5.3",
        "igraph>=0.10.4",
        "leidenalg>=0.9.1",
        "pydot>=1.4.2",
        "scikit-misc>=0.1.4",
        
        # Visualization
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        
        # Jupyter & Notebook
        "nbformat>=5.9.0",
        "nbconvert>=7.3.0",
        "ipykernel>=6.25.0",
        "jupyter-core>=5.3.0",
        "jupyter-client>=8.2.0",
        "ipython>=8.12.0",
        "comm>=0.1.3",
        "debugpy>=1.6.7",
        "matplotlib-inline>=0.1.6",
        "nest-asyncio>=1.5.6",
        "psutil>=5.9.5",
        "pyzmq>=25.1.0",
        "tornado>=6.3.2",
        "traitlets>=5.9.0",
        
        # Web Scraping & Processing
        "beautifulsoup4>=4.12.0",
        "requests-html>=0.10.0",
        "PyMuPDF>=1.22.3",
        "bleach>=6.0.0",
        "defusedxml>=0.7.1",
        "jinja2>=3.1.2",
        "jupyterlab-pygments>=0.2.2",
        "markupsafe>=2.1.3",
        "mistune>=3.0.1",
        "nbclient>=0.8.0",
        "pandocfilters>=1.5.0",
        "pygments>=2.15.1",
        "tinycss2>=1.2.1",
        
        # Docker for OpenHands integration
        "docker>=6.0.0",
        
        # Additional utilities
        "wget>=3.2",
        "sympy>=1.11.1",
        "pillow>=9.3.0",
        "loguru>=0.6.0",
        "tqdm>=4.65.0",
        
        # LLM API Support
        "openai>=1.0.0",
        "anthropic>=0.7.0",
        
        # Search & RAG
        "PyGithub>=1.59.0",
        "google-search-results>=2.4.2",
        
        # Additional dependencies
        "torch-geometric>=2.3.0",
        "rdkit-pypi>=2022.9.1",
        "mygene>=3.2.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.1.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
        ]
    },
    python_requires=">=3.8,<3.12",
    author="BioForge Team",
    author_email="bioforge@example.com",
    description="Open-Ended Autonomous Design of Computational Methods for Single-Cell Omics via Multi-Agent Collaboration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gersteinlab/scAgents",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 