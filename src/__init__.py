"""
Multi-LLM System - Democratizing AI evaluation and selection.
"""
from .providers import (
    BaseLLMProvider,
    LLMResponse,
    HuggingFaceProvider,
    GeminiProvider,
    GroqProvider,
    OllamaProvider,
    OpenRouterProvider
)
from .router import MultiLLMRouter, UseCase
from .evaluator import ResponseEvaluator

__version__ = "1.0.0"

__all__ = [
    'BaseLLMProvider',
    'LLMResponse',
    'HuggingFaceProvider',
    'GeminiProvider',
    'GroqProvider',
    'OllamaProvider',
    'OpenRouterProvider',
    'MultiLLMRouter',
    'UseCase',
    'ResponseEvaluator'
]
