"""
Base provider abstract class for Multi-LLM System.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class LLMResponse:
    """Standardized response format for all LLM providers."""
    model_name: str
    content: str
    latency: float  # in seconds
    tokens_used: Optional[int] = None
    estimated_cost: float = 0.0  # in USD
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if the response was successful."""
        return self.error is None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the provider.
        
        Args:
            api_key: API key for the provider (if required)
        """
        self.api_key = api_key
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name."""
        pass
    
    @abstractmethod
    async def generate_async(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a response asynchronously.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse object with model output and metadata
        """
        pass
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a response synchronously.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse object with model output and metadata
        """
        import asyncio
        return asyncio.run(self.generate_async(prompt, **kwargs))
    
    def _measure_latency(self, func):
        """Decorator to measure function execution time."""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            latency = time.time() - start_time
            return result, latency
        return wrapper
