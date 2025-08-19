#!/usr/bin/env python3
"""
Environment Configuration Setup Script
Helps users configure their LLM API keys and other settings
"""

import os
import json
from pathlib import Path

def create_env_template():
    """Create .env template file"""
    env_template = """# =============================================================================
# CellForge Environment Configuration
# =============================================================================

# =============================================================================
# LLM Provider Configuration
# =============================================================================

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1

# Anthropic Configuration  
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# DeepSeek Configuration
DEEPSEEK_API_KEY=sk-a54605f6118b48dab8b6e5b83faec86e

# Llama API Configuration
LLAMA_API_KEY=your_llama_api_key_here

# Qwen Configuration
QWEN_API_KEY=your_qwen_api_key_here

# Custom API Configuration
CUSTOM_API_KEY=your_custom_api_key_here
CUSTOM_API_BASE=https://your-custom-api-endpoint.com/v1

# =============================================================================
# Model Configuration
# =============================================================================

# Default Model Selection
# Options: gpt-4, gpt-4o, gpt-4o-mini, gpt-3.5-turbo (OpenAI)
#          claude-3-opus, claude-3-sonnet, claude-3-haiku (Anthropic)
#          deepseek-r1, deepseek-chat (DeepSeek)
#          llama-3.1-8b, llama-3.1-70b (Llama)
#          qwen-turbo, qwen-plus, qwen-max (Qwen)
#          custom-model-name (Custom)
MODEL_NAME=deepseek-reasoner

# Model Parameters
TEMPERATURE=0.7
MAX_TOKENS=4096
TOP_P=0.9

# =============================================================================
# Database Configuration
# =============================================================================

# Qdrant Vector Database
QDRANT_URL=localhost
QDRANT_PORT=6333

# =============================================================================
# Application Configuration
# =============================================================================

# Logging Level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# Output Directory
OUTPUT_DIR=results/

# Dataset Path
DATASET_PATH=cellforge/data/datasets/

# =============================================================================
# Advanced Configuration
# =============================================================================

# Timeout Settings (seconds)
REQUEST_TIMEOUT=30
CONNECTION_TIMEOUT=10

# Retry Settings
MAX_RETRIES=3
RETRY_DELAY=1

# Cache Settings
ENABLE_CACHE=true
CACHE_TTL=3600
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(env_template)
        print("✅ Created .env template file")
        print("📝 Please edit .env file and add your API keys")
    else:
        print("⚠️  .env file already exists")

def validate_env_config():
    """Validate current environment configuration"""
    print("=== Environment Configuration Validation ===\n")
    
    # Check LLM API keys
    llm_providers = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY", 
        "DeepSeek": "DEEPSEEK_API_KEY",
        "Llama": "LLAMA_API_KEY",
        "Qwen": "QWEN_API_KEY",
        "Custom": "CUSTOM_API_KEY"
    }
    
    configured_providers = []
    for provider, key_name in llm_providers.items():
        api_key = os.getenv(key_name)
        if api_key and api_key != "your_openai_api_key_here":
            configured_providers.append(provider)
            print(f"✅ {provider}: Configured")
        else:
            print(f"❌ {provider}: Not configured")
    
    # Check model configuration
    model_name = os.getenv("MODEL_NAME", "deepseek-r1")
    print(f"\n📋 Model Configuration:")
    print(f"   Model: {model_name}")
    print(f"   Temperature: {os.getenv('TEMPERATURE', '0.7')}")
    print(f"   Max Tokens: {os.getenv('MAX_TOKENS', '4096')}")
    
    # Check database configuration
    print(f"\n🗄️  Database Configuration:")
    print(f"   Qdrant URL: {os.getenv('QDRANT_URL', 'localhost')}")
    print(f"   Qdrant Port: {os.getenv('QDRANT_PORT', '6333')}")
    
    if configured_providers:
        print(f"\n✅ Configured LLM providers: {', '.join(configured_providers)}")
        return True
    else:
        print(f"\n❌ No LLM providers configured!")
        print("   Please add your API keys to .env file")
        return False

def test_llm_interface():
    """Test LLM interface with current configuration"""
    print("\n=== LLM Interface Test ===\n")
    
    try:
        from cellforge.llm import LLMInterface
        llm = LLMInterface()
        
        # Get configuration status
        config = llm.get_config_status()
        print("Configuration Status:")
        for provider, status in config.items():
            if provider.endswith("_configured"):
                provider_name = provider.replace("_configured", "").title()
                print(f"   {provider_name}: {'✅' if status else '❌'}")
        
        # Test simple generation
        print("\nTesting LLM generation...")
        test_prompt = "Hello, please respond with a simple greeting."
        test_system = "You are a helpful assistant."
        
        response = llm.generate(test_prompt, test_system)
        print("✅ LLM generation successful!")
        print(f"Response type: {type(response)}")
        if isinstance(response, dict):
            print(f"Response keys: {list(response.keys())}")
            if "content" in response:
                print(f"Content preview: {response['content'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM interface test failed: {str(e)}")
        return False

def main():
    """Main setup function"""
    print("🔧 CellForge Environment Setup\n")
    
    # Create .env template if it doesn't exist
    create_env_template()
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded environment variables from .env file")
    except ImportError:
        print("⚠️  python-dotenv not installed, using system environment variables")
    except Exception as e:
        print(f"⚠️  Could not load .env file: {e}")
    
    # Validate configuration
    is_valid = validate_env_config()
    
    if is_valid:
        # Test LLM interface
        test_llm_interface()
    
    print("\n=== Setup Complete ===")
    print("💡 If you need to configure API keys, edit the .env file")
    print("💡 For help, see the documentation or run this script again")

if __name__ == "__main__":
    main() 