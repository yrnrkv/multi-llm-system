"""
Ollama local provider for Multi-LLM System.
Completely free, runs models locally.
"""
import httpx
import time
from typing import Optional
from .base import BaseLLMProvider, LLMResponse


class OllamaProvider(BaseLLMProvider):
    """Provider for Ollama (local inference)."""
    
    DEFAULT_MODELS = [
        "llama3",
        "mistral",
        "phi3"
    ]
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = None):
        """
        Initialize Ollama provider.
        
        Args:
            base_url: Ollama server URL
            model: Model to use (default: llama3)
        """
        super().__init__(None)  # No API key needed
        self.base_url = base_url
        self.model = model or self.DEFAULT_MODELS[0]
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return f"ollama/{self.model}"
    
    async def generate_async(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a response using Ollama API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse object
        """
        start_time = time.time()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                
                latency = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("response", "")
                    
                    return LLMResponse(
                        model_name=f"ollama/{self.model}",
                        content=content,
                        latency=latency,
                        estimated_cost=0.0  # Completely free (local)
                    )
                else:
                    return LLMResponse(
                        model_name=f"ollama/{self.model}",
                        content="",
                        latency=latency,
                        error=f"Ollama API error: {response.status_code}"
                    )
                    
        except httpx.ConnectError:
            latency = time.time() - start_time
            return LLMResponse(
                model_name=f"ollama/{self.model}",
                content="",
                latency=latency,
                error="Cannot connect to Ollama. Is it running? Install from https://ollama.ai/"
            )
        except Exception as e:
            latency = time.time() - start_time
            return LLMResponse(
                model_name=f"ollama/{self.model}",
                content="",
                latency=latency,
                error=f"Request failed: {str(e)}"
            )
