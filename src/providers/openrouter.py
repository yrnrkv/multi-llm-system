"""
OpenRouter API provider for Multi-LLM System.
Provides access to multiple free models.
"""
import httpx
import time
from typing import Optional
from .base import BaseLLMProvider, LLMResponse


class OpenRouterProvider(BaseLLMProvider):
    """Provider for OpenRouter API."""
    
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    # Free models available on OpenRouter
    DEFAULT_MODELS = [
        "meta-llama/llama-3-8b-instruct:free",
        "google/gemma-7b-it:free",
        "mistralai/mistral-7b-instruct:free"
    ]
    
    def __init__(self, api_key: Optional[str] = None, model: str = None):
        """
        Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key
            model: Model to use (default: llama-3-8b-instruct:free)
        """
        super().__init__(api_key)
        self.model = model or self.DEFAULT_MODELS[0]
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model
    
    async def generate_async(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a response using OpenRouter API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
        """
        start_time = time.time()
        
        if not self.api_key:
            return LLMResponse(
                model_name=self.model,
                content="",
                latency=0.0,
                error="OpenRouter API key not provided"
            )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/multi-llm-system",
            "X-Title": "Multi-LLM System"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload
                )
                
                latency = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    tokens_used = result.get("usage", {}).get("total_tokens")
                    
                    return LLMResponse(
                        model_name=self.model,
                        content=content,
                        latency=latency,
                        tokens_used=tokens_used,
                        estimated_cost=0.0  # Free models
                    )
                else:
                    error_msg = f"API error: {response.status_code}"
                    try:
                        error_detail = response.json()
                        error_msg += f" - {error_detail}"
                    except:
                        pass
                    
                    return LLMResponse(
                        model_name=self.model,
                        content="",
                        latency=latency,
                        error=error_msg
                    )
                    
        except Exception as e:
            latency = time.time() - start_time
            return LLMResponse(
                model_name=self.model,
                content="",
                latency=latency,
                error=f"Request failed: {str(e)}"
            )
