#!/usr/bin/env python3
"""
Debug model routing
"""

import os
from cellforge.llm import LLMInterface

def debug_model_routing():
    """Debug model routing logic"""
    print("=== Model Routing Debug ===\n")
    
    # Set environment variables
    os.environ["DEEPSEEK_API_KEY"] = "sk-a54605f6118b48dab8b6e5b83faec86e"
    os.environ["MODEL_NAME"] = "deepseek-reasoner"
    
    print("Environment variables:")
    print(f"DEEPSEEK_API_KEY: {os.getenv('DEEPSEEK_API_KEY')[:8]}...")
    print(f"MODEL_NAME: {os.getenv('MODEL_NAME')}")
    
    # Create LLM interface
    llm = LLMInterface()
    
    print(f"\nLLM Interface Configuration:")
    print(f"Model name: {llm.model_name}")
    print(f"DeepSeek configured: {llm.deepseek_api_key is not None}")
    
    # Test model routing logic
    test_models = [
        "deepseek-r1",
        "deepseek-reasoner", 
        "deepseek-chat",
        "deepseek-v3",
        "gpt-4",
        "claude-3"
    ]
    
    print(f"\nModel Routing Test:")
    for model in test_models:
        print(f"\nTesting model: {model}")
        
        # Simulate the routing logic
        if model.startswith("gpt"):
            print(f"  -> Would route to OpenAI")
        elif model.startswith("claude"):
            print(f"  -> Would route to Anthropic")
        elif model.startswith("deepseek"):
            if model == "deepseek-r1":
                mapped_model = "deepseek-reasoner"
            elif model == "deepseek-chat" or model == "deepseek-v3":
                mapped_model = "deepseek-chat"
            else:
                mapped_model = model
            print(f"  -> Would route to DeepSeek with model: {mapped_model}")
        elif model.startswith("llama"):
            print(f"  -> Would route to Llama")
        elif model.startswith("qwen"):
            print(f"  -> Would route to Qwen")
        else:
            print(f"  -> Would fallback to OpenAI")
    
    # Test actual generation
    print(f"\nActual Generation Test:")
    try:
        response = llm.generate("Hello", "You are a helpful assistant.")
        print(f"✅ Generation successful!")
        print(f"Provider: {response.get('provider', 'Unknown')}")
        print(f"Model: {response.get('model', 'Unknown')}")
        print(f"Content: {response.get('content', 'No content')[:100]}...")
    except Exception as e:
        print(f"❌ Generation failed: {e}")

if __name__ == "__main__":
    debug_model_routing() 