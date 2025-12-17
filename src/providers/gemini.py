"""
Google Gemini API provider for Multi-LLM System.
"""
import time
from typing import Optional
from .base import BaseLLMProvider, LLMResponse


class GeminiProvider(BaseLLMProvider):
    """Provider for Google Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-pro"):
        """
        Initialize Gemini provider.
        
        Args:
            api_key: Google API key
            model: Model to use (default: gemini-pro)
        """
        super().__init__(api_key)
        self.model = model
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                if self.api_key:
                    genai.configure(api_key=self.api_key)
                    self._client = genai.GenerativeModel(self.model)
            except ImportError:
                pass
        return self._client
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return self.model
    
    async def generate_async(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a response using Gemini API.
        
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
                error="Gemini API key not provided"
            )
        
        client = self._get_client()
        if client is None:
            return LLMResponse(
                model_name=self.model,
                content="",
                latency=0.0,
                error="google-generativeai package not installed"
            )
        
        try:
            # Generate content
            response = client.generate_content(prompt)
            latency = time.time() - start_time
            
            # Extract text from response
            if hasattr(response, 'text'):
                content = response.text
            elif hasattr(response, 'parts'):
                content = ''.join([part.text for part in response.parts])
            else:
                content = str(response)
            
            return LLMResponse(
                model_name=self.model,
                content=content,
                latency=latency,
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
