"""
Basic usage example for Multi-LLM System.
Demonstrates how to use the system programmatically.
"""
import os
import asyncio
from dotenv import load_dotenv
from src.providers import (
    HuggingFaceProvider,
    GeminiProvider,
    GroqProvider,
    OllamaProvider,
    OpenRouterProvider
)
from src.router import MultiLLMRouter, UseCase
from src.evaluator import ResponseEvaluator

# Load environment variables
load_dotenv()


async def example_single_provider():
    """Example: Using a single provider."""
    print("=" * 60)
    print("Example 1: Single Provider")
    print("=" * 60)
    
    # Initialize a provider (using Groq as example)
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("⚠️ GROQ_API_KEY not found, skipping this example")
        return
    
    provider = GroqProvider(api_key=groq_key)
    
    # Generate a response
    prompt = "Explain what artificial intelligence is in simple terms."
    print(f"\nPrompt: {prompt}\n")
    
    response = await provider.generate_async(prompt)
    
    if response.success:
        print(f"Model: {response.model_name}")
        print(f"Latency: {response.latency:.2f}s")
        print(f"\nResponse:\n{response.content}\n")
    else:
        print(f"Error: {response.error}")


async def example_compare_all():
    """Example: Compare all providers."""
    print("\n" + "=" * 60)
    print("Example 2: Compare All Providers")
    print("=" * 60)
    
    # Initialize router
    router = MultiLLMRouter()
    
    # Register available providers
    providers_count = 0
    
    if os.getenv("HUGGINGFACE_API_KEY"):
        router.register_provider("huggingface", 
                                HuggingFaceProvider(api_key=os.getenv("HUGGINGFACE_API_KEY")))
        providers_count += 1
    
    if os.getenv("GEMINI_API_KEY"):
        router.register_provider("gemini", 
                                GeminiProvider(api_key=os.getenv("GEMINI_API_KEY")))
        providers_count += 1
    
    if os.getenv("GROQ_API_KEY"):
        router.register_provider("groq", 
                                GroqProvider(api_key=os.getenv("GROQ_API_KEY")))
        providers_count += 1
    
    if os.getenv("OPENROUTER_API_KEY"):
        router.register_provider("openrouter", 
                                OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY")))
        providers_count += 1
    
    # Always register Ollama
    router.register_provider("ollama", OllamaProvider())
    providers_count += 1
    
    print(f"\n✅ Registered {providers_count} providers")
    
    # Query all providers
    prompt = "What are three benefits of exercise?"
    print(f"\nPrompt: {prompt}\n")
    print("Querying all providers in parallel...\n")
    
    responses = await router.query_all_async(prompt)
    
    # Evaluate and compare
    comparison = ResponseEvaluator.compare_responses(responses)
    
    # Display results
    print(f"Results: {comparison['successful_models']}/{comparison['total_models']} successful")
    
    if comparison.get('fastest_model'):
        print(f"Fastest: {comparison['fastest_model']} ({comparison['fastest_latency']:.2f}s)")
    
    if comparison.get('most_readable_model'):
        print(f"Most Readable: {comparison['most_readable_model']}")
    
    print("\n" + "-" * 60)
    print("Individual Responses:")
    print("-" * 60)
    
    for name, response in responses.items():
        evaluation = comparison['evaluations'][name]
        print(f"\n🤖 {name.upper()}")
        
        if evaluation['success']:
            print(f"   Model: {response.model_name}")
            print(f"   Speed: {evaluation['speed_rating']} ({evaluation['latency']:.2f}s)")
            print(f"   Response: {response.content[:150]}...")
        else:
            print(f"   ❌ Error: {evaluation['error']}")


async def example_use_case_routing():
    """Example: Use case-based routing."""
    print("\n" + "=" * 60)
    print("Example 3: Use Case-Based Routing")
    print("=" * 60)
    
    # Initialize router with at least one provider
    router = MultiLLMRouter()
    
    if os.getenv("GROQ_API_KEY"):
        router.register_provider("groq", GroqProvider(api_key=os.getenv("GROQ_API_KEY")))
    elif os.getenv("GEMINI_API_KEY"):
        router.register_provider("gemini", GeminiProvider(api_key=os.getenv("GEMINI_API_KEY")))
    else:
        router.register_provider("ollama", OllamaProvider())
    
    # Healthcare use case
    prompt = "What are common symptoms of diabetes?"
    print(f"\nHealthcare Query: {prompt}")
    print(f"Recommended models: {router.USE_CASE_PREFERENCES[UseCase.HEALTHCARE]}")
    
    response = await router.query_best_for_use_case_async(prompt, UseCase.HEALTHCARE)
    
    if response.success:
        print(f"\n✅ Response from: {response.model_name}")
        print(f"   Latency: {response.latency:.2f}s")
        print(f"   Response: {response.content[:200]}...\n")
    else:
        print(f"\n❌ Error: {response.error}\n")


async def example_accessibility_features():
    """Example: Accessibility and readability evaluation."""
    print("\n" + "=" * 60)
    print("Example 4: Accessibility & Readability")
    print("=" * 60)
    
    # Sample response text
    sample_text = """
    Regular physical activity can help you maintain a healthy weight and reduce
    your risk of chronic diseases. It strengthens your heart and improves circulation.
    Exercise also boosts your mood by releasing endorphins, which are natural mood lifters.
    """
    
    print("\nEvaluating text readability...")
    readability = ResponseEvaluator.calculate_readability(sample_text)
    
    print(f"\n📖 Readability Analysis:")
    print(f"   Reading Level: {readability.get('interpretation', 'N/A')}")
    print(f"   Word Count: {readability.get('word_count', 0)}")
    print(f"   Sentence Count: {readability.get('sentence_count', 0)}")
    
    if 'flesch_reading_ease' in readability:
        print(f"   Flesch Reading Ease: {readability['flesch_reading_ease']}")
        print(f"   Flesch-Kincaid Grade: {readability['flesch_kincaid_grade']}")


async def main():
    """Run all examples."""
    print("\n🤖 Multi-LLM System - Basic Usage Examples")
    print("=" * 60)
    
    # Run examples
    await example_single_provider()
    await example_compare_all()
    await example_use_case_routing()
    await example_accessibility_features()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    print("\n💡 Tips:")
    print("   - Set up your API keys in a .env file")
    print("   - Run 'streamlit run app.py' for the web interface")
    print("   - Check README.md for more information")
    print()


if __name__ == "__main__":
    asyncio.run(main())
