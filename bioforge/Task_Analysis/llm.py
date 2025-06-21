import os
import json
import requests
from typing import Dict, Any, Optional
import openai
from anthropic import Anthropic
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMInterface:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.llama_api_key = os.getenv("LLAMA_API_KEY")
        self.qwen_api_key = os.getenv("QWEN_API_KEY")
        self.custom_api_url = os.getenv("CUSTOM_API_URL")
        self.custom_api_key = os.getenv("CUSTOM_API_KEY")
        
        self.model_name = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
        
        # Validate environment variables
        self._validate_environment()
        
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        if self.anthropic_api_key:
            self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)
            
    def _validate_environment(self):
        """Validate environment variable configuration"""
        configured_providers = []
        
        if self.openai_api_key:
            configured_providers.append("OpenAI")
        if self.anthropic_api_key:
            configured_providers.append("Anthropic")
        if self.deepseek_api_key:
            configured_providers.append("DeepSeek")
        if self.llama_api_key:
            configured_providers.append("Llama")
        if self.qwen_api_key:
            configured_providers.append("Qwen")
        if self.custom_api_url and self.custom_api_key:
            configured_providers.append("Custom")
            
        if not configured_providers:
            logger.warning("No LLM API keys configured. Please set at least one API key.")
        else:
            logger.info(f"Configured LLM providers: {', '.join(configured_providers)}")
            
    def get_config_status(self) -> Dict[str, Any]:
        """Get configuration status"""
        return {
            "openai_configured": bool(self.openai_api_key),
            "anthropic_configured": bool(self.anthropic_api_key),
            "deepseek_configured": bool(self.deepseek_api_key),
            "llama_configured": bool(self.llama_api_key),
            "qwen_configured": bool(self.qwen_api_key),
            "custom_configured": bool(self.custom_api_url and self.custom_api_key),
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
            
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                model: Optional[str] = None, temperature: Optional[float] = None,
                max_tokens: Optional[int] = None) -> Dict[str, Any]:
        model = model or self.model_name
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        try:
            if model.startswith("gpt"):
                return self._generate_openai(prompt, system_prompt, model, temperature, max_tokens)
            elif model.startswith("claude"):
                return self._generate_anthropic(prompt, system_prompt, model, temperature, max_tokens)
            elif model.startswith("deepseek"):
                return self._generate_deepseek(prompt, system_prompt, model, temperature, max_tokens)
            elif model.startswith("llama"):
                return self._generate_llama(prompt, system_prompt, model, temperature, max_tokens)
            elif model.startswith("qwen"):
                return self._generate_qwen(prompt, system_prompt, model, temperature, max_tokens)
            elif self.custom_api_url:
                return self._generate_custom(prompt, system_prompt, model, temperature, max_tokens)
            else:
                raise ValueError(f"Unsupported model: {model}")
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")
            
    def _generate_openai(self, prompt: str, system_prompt: str, model: str,
                        temperature: float, max_tokens: int) -> Dict[str, Any]:
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
        return json.loads(response.choices[0].message.content)
        
    def _generate_anthropic(self, prompt: str, system_prompt: str, model: str,
                          temperature: float, max_tokens: int) -> Dict[str, Any]:
        response = self.anthropic_client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return json.loads(response.content[0].text)
        
    def _generate_deepseek(self, prompt: str, system_prompt: str, model: str,
                          temperature: float, max_tokens: int) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post("https://api.deepseek.com/v1/chat/completions",
                               headers=headers, json=data)
        return json.loads(response.json()["choices"][0]["message"]["content"])
        
    def _generate_llama(self, prompt: str, system_prompt: str, model: str,
                       temperature: float, max_tokens: int) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.llama_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post("https://api.llama-api.com/v1/chat/completions",
                               headers=headers, json=data)
        return json.loads(response.json()["choices"][0]["message"]["content"])
        
    def _generate_qwen(self, prompt: str, system_prompt: str, model: str,
                      temperature: float, max_tokens: int) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.qwen_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post("https://api.qwen.ai/v1/chat/completions",
                               headers=headers, json=data)
        return json.loads(response.json()["choices"][0]["message"]["content"])
        
    def _generate_custom(self, prompt: str, system_prompt: str, model: str,
                        temperature: float, max_tokens: int) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.custom_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post(self.custom_api_url, headers=headers, json=data)
        return json.loads(response.json()["choices"][0]["message"]["content"]) 