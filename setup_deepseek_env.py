#!/usr/bin/env python3
"""
Setup script for DeepSeek API environment variables
"""

import os
import sys
from pathlib import Path

def setup_deepseek_env():
    """Setup DeepSeek API environment variables"""
    print("=== DeepSeek API Setup ===")
    print("This script will help you set up your DeepSeek API key.")
    print()
    
    # Check if API key is already set
    current_key = os.getenv("DEEPSEEK_API_KEY")
    if current_key and current_key != "your_deepseek_api_key_here":
        print(f"✅ DeepSeek API key is already configured")
        print(f"Current key: {current_key[:10]}...{current_key[-4:]}")
        return True
    
    print("❌ DeepSeek API key not configured")
    print()
    print("To get a DeepSeek API key:")
    print("1. Visit https://platform.deepseek.com/")
    print("2. Sign up or log in to your account")
    print("3. Go to API Keys section")
    print("4. Create a new API key")
    print()
    
    # Ask user for API key
    api_key = input("Please enter your DeepSeek API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("⚠️ No API key provided. You can set it later using:")
        print("   $env:DEEPSEEK_API_KEY = 'your_api_key_here'")
        return False
    
    # Set environment variable for current session
    os.environ["DEEPSEEK_API_KEY"] = api_key
    print(f"✅ DeepSeek API key set for current session")
    
    # Create .env file if it doesn't exist
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(f"DEEPSEEK_API_KEY={api_key}\n")
        print("✅ Created .env file with API key")
    else:
        # Update existing .env file
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Check if DEEPSEEK_API_KEY already exists
            updated = False
            for i, line in enumerate(lines):
                if line.startswith("DEEPSEEK_API_KEY="):
                    lines[i] = f"DEEPSEEK_API_KEY={api_key}\n"
                    updated = True
                    break
            
            if not updated:
                lines.append(f"DEEPSEEK_API_KEY={api_key}\n")
            
            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            
            print("✅ Updated .env file with API key")
        except Exception as e:
            print(f"⚠️ Could not update .env file: {e}")
    
    return True

def test_connection():
    """Test DeepSeek API connection"""
    print("\n=== Testing DeepSeek Connection ===")
    
    try:
        from cellforge.llm import LLMInterface
        
        llm = LLMInterface()
        config = llm.get_config_status()
        
        if not config["deepseek_configured"]:
            print("❌ DeepSeek API key not configured")
            return False
        
        print("✅ DeepSeek API key is configured")
        
        # Test simple generation
        print("Testing API connection...")
        test_prompt = "Hello, please respond with 'Connection successful'"
        
        response = llm.generate(test_prompt, model="deepseek-chat")
        print("✅ Connection successful!")
        print(f"Response: {response.get('content', '')}")
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def main():
    """Main function"""
    print("DeepSeek API Setup and Test")
    print("=" * 30)
    
    # Setup environment
    if setup_deepseek_env():
        # Test connection
        test_connection()
    else:
        print("\n⚠️ Please configure your API key and run this script again")
    
    print("\n=== Setup Complete ===")
    print("You can now run your CellForge applications with DeepSeek support!")

if __name__ == "__main__":
    main()
