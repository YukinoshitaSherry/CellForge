import os
from typing import Dict, Any, Optional
import openai
from anthropic import Anthropic
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMInterface:    
    def __init__(self):
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Validate environment variables
        self._validate_environment()
        
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        if self.anthropic_api_key:
            self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
            
        
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4096000"))
        
    def _validate_environment(self):
        """Validate environment variable configuration"""
        if not self.openai_api_key and not self.anthropic_api_key:
            logger.warning("No LLM API keys configured. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        
        if self.openai_api_key:
            logger.info("OpenAI API key configured")
        if self.anthropic_api_key:
            logger.info("Anthropic API key configured")
            
    def get_config_status(self) -> Dict[str, Any]:
        """Get configuration status"""
        return {
            "openai_configured": bool(self.openai_api_key),
            "anthropic_configured": bool(self.anthropic_api_key),
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None,
                model: Optional[str] = None,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Unified LLM interface
        
        Args:
            prompt: user prompt
            system_prompt: system prompt (optional)
            model: model name (optional)
            temperature: temperature parameter (optional)
            max_tokens: maximum token number (optional)
            
        Returns:
            generated response content
        """
        
        model = model or self.model_name
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        try:
            
            if self.openai_api_key:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return {
                    "content": response.choices[0].message.content,
                    "model": model,
                    "provider": "openai"
                }
                
            
            elif self.anthropic_api_key:
                response = self.anthropic_client.messages.create(
                    model=model,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return {
                    "content": response.content[0].text,
                    "model": model,
                    "provider": "anthropic"
                }
                
            else:
                raise ValueError("No LLM API key configured")
                
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")
            
    def parse_json_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse JSON formatted response
        
        Args:
            response: LLM generated response
            
        Returns:
            parsed JSON object
        """
        try:
            content = response["content"]
            
            return json.loads(content)
        except json.JSONDecodeError:
            
            import re
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                raise ValueError("Response does not contain valid JSON") 