"""
Hugging Face Inference API provider for Multi-LLM System.
"""
import httpx
import time
from typing import Optional
from .base import BaseLLMProvider, LLMResponse


class HuggingFaceProvider(BaseLLMProvider):
    """Provider for Hugging Face Inference API."""
    
    BASE_URL = "https://api-inference.huggingface.co/models/"
    
    # Free models available on Hugging Face
    DEFAULT_MODELS = [
        "mistralai/Mistral-7B-Instruct-v0.1",
        "meta-llama/Llama-2-7b-chat-hf",
        "HuggingFaceH4/zephyr-7b-beta"
    ]
    
    def __init__(self, api_key: Optional[str] = None, model: str = None):
        """
        Initialize HuggingFace provider.
        
        Args:
            api_key: HuggingFace API token
            model: Model to use (defaults to Mistral-7B-Instruct)
        """
        super().__init__(api_key)
        self.model = model or self.DEFAULT_MODELS[0]
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model
    
    async def generate_async(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a response using HuggingFace API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (max_new_tokens, temperature, etc.)
            
        Returns:
            LLMResponse object
        """
        start_time = time.time()
        
        if not self.api_key:
            return LLMResponse(
                model_name=self.model,
                content="",
                latency=0.0,
                error="HuggingFace API key not provided"
            )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 0.7),
                "return_full_text": False
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.BASE_URL}{self.model}",
                    headers=headers,
                    json=payload
                )
                
                latency = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle different response formats
                    if isinstance(result, list) and len(result) > 0:
                        content = result[0].get("generated_text", "")
                    elif isinstance(result, dict):
                        content = result.get("generated_text", "")
                    else:
                        content = str(result)
                    
                    return LLMResponse(
                        model_name=self.model,
                        content=content,
                        latency=latency,
                        estimated_cost=0.0  # Free tier
                    )
                else:
                    error_msg = f"API error: {response.status_code}"
                    try:
                        error_detail = response.json()
                        error_msg += f" - {error_detail}"
                    except (ValueError, KeyError):
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
