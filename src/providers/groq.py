"""
Groq API provider for Multi-LLM System.
Known for extremely fast inference speeds.
"""
import time
from typing import Optional
from .base import BaseLLMProvider, LLMResponse


class GroqProvider(BaseLLMProvider):
    """Provider for Groq API (very fast inference)."""
    
    # Free models available on Groq
    DEFAULT_MODELS = [
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma-7b-it"
    ]
    
    def __init__(self, api_key: Optional[str] = None, model: str = None):
        """
        Initialize Groq provider.
        
        Args:
            api_key: Groq API key
            model: Model to use (default: llama3-8b-8192)
        """
        super().__init__(api_key)
        self.model = model or self.DEFAULT_MODELS[0]
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of Groq client."""
        if self._client is None and self.api_key:
            try:
                from groq import AsyncGroq
                self._client = AsyncGroq(api_key=self.api_key)
            except ImportError:
                pass
        return self._client
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model
    
    async def generate_async(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a response using Groq API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (max_tokens, temperature, etc.)
            
        Returns:
            LLMResponse object
        """
        start_time = time.time()
        
        if not self.api_key:
            return LLMResponse(
                model_name=self.model,
                content="",
                latency=0.0,
                error="Groq API key not provided"
            )
        
        client = self._get_client()
        if client is None:
            return LLMResponse(
                model_name=self.model,
                content="",
                latency=0.0,
                error="groq package not installed"
            )
        
        try:
            # Create chat completion
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", 0.7)
            )
            
            latency = time.time() - start_time
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            return LLMResponse(
                model_name=self.model,
                content=content,
                latency=latency,
                tokens_used=tokens_used,
                estimated_cost=0.0  # Free tier
            )
            
        except Exception as e:
            latency = time.time() - start_time
            return LLMResponse(
                model_name=self.model,
                content="",
                latency=latency,
                error=f"Request failed: {str(e)}"
            )
