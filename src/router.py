"""
Multi-LLM Router for intelligent model selection and comparison.
"""
import asyncio
from typing import List, Dict, Optional
from enum import Enum
from .providers.base import BaseLLMProvider, LLMResponse


class UseCase(Enum):
    """Use case categories for intelligent routing."""
    HEALTHCARE = "healthcare"
    ACCESSIBILITY = "accessibility"
    GENERAL = "general"
    COST_SENSITIVE = "cost_sensitive"


class MultiLLMRouter:
    """Router for managing multiple LLM providers."""
    
    # Provider preferences by use case
    USE_CASE_PREFERENCES = {
        UseCase.HEALTHCARE: ["gemini", "groq", "huggingface"],  # Prefer reliable, accurate models
        UseCase.ACCESSIBILITY: ["groq", "gemini", "openrouter"],  # Prefer fast, clear responses
        UseCase.GENERAL: ["groq", "ollama", "gemini"],  # Balance of speed and quality
        UseCase.COST_SENSITIVE: ["ollama", "groq", "openrouter"]  # Prefer free/local models
    }
    
    def __init__(self):
        """Initialize the router."""
        self.providers: Dict[str, BaseLLMProvider] = {}
    
    def register_provider(self, name: str, provider: BaseLLMProvider):
        """
        Register a provider with the router.
        
        Args:
            name: Unique name for the provider
            provider: Provider instance
        """
        self.providers[name] = provider
    
    def get_provider(self, name: str) -> Optional[BaseLLMProvider]:
        """
        Get a provider by name.
        
        Args:
            name: Provider name
            
        Returns:
            Provider instance or None
        """
        return self.providers.get(name)
    
    def list_providers(self) -> List[str]:
        """
        List all registered provider names.
        
        Returns:
            List of provider names
        """
        return list(self.providers.keys())
    
    async def query_all_async(self, prompt: str, **kwargs) -> Dict[str, LLMResponse]:
        """
        Query all registered providers in parallel.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for providers
            
        Returns:
            Dictionary mapping provider names to responses
        """
        tasks = []
        provider_names = []
        
        for name, provider in self.providers.items():
            tasks.append(provider.generate_async(prompt, **kwargs))
            provider_names.append(name)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = {}
        for name, response in zip(provider_names, responses):
            if isinstance(response, Exception):
                # Handle exceptions
                results[name] = LLMResponse(
                    model_name=name,
                    content="",
                    latency=0.0,
                    error=f"Exception: {str(response)}"
                )
            else:
                results[name] = response
        
        return results
    
    def query_all(self, prompt: str, **kwargs) -> Dict[str, LLMResponse]:
        """
        Query all registered providers (synchronous version).
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for providers
            
        Returns:
            Dictionary mapping provider names to responses
        """
        return asyncio.run(self.query_all_async(prompt, **kwargs))
    
    async def query_best_for_use_case_async(
        self,
        prompt: str,
        use_case: UseCase,
        **kwargs
    ) -> LLMResponse:
        """
        Query the best provider for a specific use case.
        
        Args:
            prompt: The input prompt
            use_case: The use case category
            **kwargs: Additional parameters for providers
            
        Returns:
            Response from the best available provider
        """
        preferences = self.USE_CASE_PREFERENCES.get(use_case, [])
        
        # Try providers in order of preference
        for provider_name in preferences:
            provider = self.providers.get(provider_name)
            if provider:
                response = await provider.generate_async(prompt, **kwargs)
                if response.success:
                    return response
        
        # Fallback: try any available provider
        for provider_name, provider in self.providers.items():
            if provider_name not in preferences:
                response = await provider.generate_async(prompt, **kwargs)
                if response.success:
                    return response
        
        # All providers failed
        return LLMResponse(
            model_name="none",
            content="",
            latency=0.0,
            error="No providers available or all providers failed"
        )
    
    def query_best_for_use_case(
        self,
        prompt: str,
        use_case: UseCase,
        **kwargs
    ) -> LLMResponse:
        """
        Query the best provider for a specific use case (synchronous version).
        
        Args:
            prompt: The input prompt
            use_case: The use case category
            **kwargs: Additional parameters for providers
            
        Returns:
            Response from the best available provider
        """
        return asyncio.run(
            self.query_best_for_use_case_async(prompt, use_case, **kwargs)
        )
    
    def get_use_case_explanation(self, use_case: UseCase) -> str:
        """
        Get an explanation for why certain models are preferred for a use case.
        
        Args:
            use_case: The use case category
            
        Returns:
            Human-readable explanation
        """
        explanations = {
            UseCase.HEALTHCARE: "For healthcare queries, we prioritize accurate and reliable models like Gemini and Groq that can provide detailed medical information.",
            UseCase.ACCESSIBILITY: "For accessibility needs, we use fast models like Groq and Gemini that provide clear, easy-to-understand responses quickly.",
            UseCase.GENERAL: "For general queries, we balance speed and quality using Groq and Ollama for fast, helpful responses.",
            UseCase.COST_SENSITIVE: "For cost-sensitive users, we prioritize free local models like Ollama and free tier services."
        }
        return explanations.get(use_case, "General purpose AI assistance.")
