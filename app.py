"""
Streamlit Web Interface for Multi-LLM System.
Accessible, user-friendly interface for elderly and disabled users.
"""
import streamlit as st
import os
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

# Page configuration
st.set_page_config(
    page_title="Multi-LLM System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for accessibility
st.markdown("""
    <style>
    /* Larger, more readable text */
    .stTextInput > label, .stSelectbox > label, .stTextArea > label {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }
    
    .stButton > button {
        font-size: 1.2rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
    }
    
    /* High contrast */
    .stTextInput > div > div > input {
        font-size: 1.1rem !important;
    }
    
    .stTextArea > div > div > textarea {
        font-size: 1.1rem !important;
    }
    
    /* Response cards */
    .response-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
    
    .error-card {
        background-color: #ffebee;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #f44336;
    }
    
    /* Metrics */
    .metric-container {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    h1 {
        font-size: 2.5rem !important;
    }
    
    h2 {
        font-size: 2rem !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'router' not in st.session_state:
    st.session_state.router = None
if 'responses' not in st.session_state:
    st.session_state.responses = None


def initialize_router():
    """Initialize the router with all available providers."""
    router = MultiLLMRouter()
    
    # Get API keys from environment
    hf_key = os.getenv("HUGGINGFACE_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Register providers that have API keys
    providers_registered = []
    
    if hf_key:
        router.register_provider("huggingface", HuggingFaceProvider(api_key=hf_key))
        providers_registered.append("HuggingFace")
    
    if gemini_key:
        router.register_provider("gemini", GeminiProvider(api_key=gemini_key))
        providers_registered.append("Gemini")
    
    if groq_key:
        router.register_provider("groq", GroqProvider(api_key=groq_key))
        providers_registered.append("Groq")
    
    if openrouter_key:
        router.register_provider("openrouter", OpenRouterProvider(api_key=openrouter_key))
        providers_registered.append("OpenRouter")
    
    # Always try to register Ollama (local)
    router.register_provider("ollama", OllamaProvider(base_url=ollama_url))
    providers_registered.append("Ollama (local)")
    
    return router, providers_registered


def display_response(name, response, evaluation):
    """Display a single response with metrics."""
    if evaluation.get("success"):
        st.markdown(f"### 🤖 {name}")
        st.markdown(f"**Model:** {response.model_name}")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Response Time", f"{evaluation['latency']:.2f}s", 
                     delta=evaluation['speed_rating'])
        with col2:
            if evaluation.get('tokens_used'):
                st.metric("Tokens Used", evaluation['tokens_used'])
        with col3:
            st.metric("Cost", f"${evaluation['cost']:.4f}")
        
        # Readability
        readability = evaluation.get('readability', {})
        if 'interpretation' in readability:
            st.info(f"📖 Readability: {readability['interpretation']}")
        
        # Response text
        st.markdown("**Response:**")
        st.markdown(f"<div class='response-card'>{response.content}</div>", 
                   unsafe_allow_html=True)
    else:
        st.markdown(f"### ❌ {name}")
        st.markdown(f"<div class='error-card'>**Error:** {evaluation.get('error', 'Unknown error')}</div>", 
                   unsafe_allow_html=True)


def main():
    """Main application."""
    
    # Header
    st.title("🤖 Multi-LLM System")
    st.markdown("### *Tech for Social Good - AI for Everyone*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This system helps you find the best AI model for your needs.
        
        **Perfect for:**
        - Healthcare information
        - Accessibility support
        - General questions
        - Cost-conscious users
        """)
        
        st.markdown("---")
        
        # Initialize router
        if st.session_state.router is None:
            with st.spinner("Setting up AI models..."):
                st.session_state.router, providers = initialize_router()
            
            st.success(f"✅ {len(providers)} providers ready!")
            for provider in providers:
                st.markdown(f"- {provider}")
        else:
            providers = st.session_state.router.list_providers()
            st.success(f"✅ {len(providers)} providers active")
        
        st.markdown("---")
        st.markdown("**Need API Keys?**")
        st.markdown("""
        - [HuggingFace](https://huggingface.co/settings/tokens)
        - [Google Gemini](https://makersuite.google.com/app/apikey)
        - [Groq](https://console.groq.com/keys)
        - [OpenRouter](https://openrouter.ai/keys)
        - [Ollama](https://ollama.ai/) (local, no key)
        """)
    
    # Main content
    router = st.session_state.router
    
    if router is None:
        st.error("Please set up at least one API key in your .env file")
        return
    
    # Use case selector
    st.subheader("1️⃣ What do you need help with?")
    use_case_options = {
        "General Questions": UseCase.GENERAL,
        "Healthcare Information": UseCase.HEALTHCARE,
        "Accessibility Support": UseCase.ACCESSIBILITY,
        "Cost-Effective Solutions": UseCase.COST_SENSITIVE
    }
    
    selected_use_case_name = st.selectbox(
        "Select your need:",
        options=list(use_case_options.keys()),
        help="Choose the category that best matches your question"
    )
    selected_use_case = use_case_options[selected_use_case_name]
    
    # Show explanation
    explanation = router.get_use_case_explanation(selected_use_case)
    st.info(f"ℹ️ {explanation}")
    
    # Prompt input
    st.subheader("2️⃣ Ask your question")
    prompt = st.text_area(
        "Type your question here:",
        height=150,
        placeholder="Example: What are the symptoms of the flu?",
        help="Ask any question - the AI will help you!"
    )
    
    # Action buttons
    st.subheader("3️⃣ Get your answer")
    col1, col2 = st.columns(2)
    
    with col1:
        single_query = st.button("🚀 Get Best Answer", 
                                 type="primary",
                                 help="Get answer from the best AI for your need",
                                 use_container_width=True)
    
    with col2:
        compare_all = st.button("📊 Compare All Models",
                               help="See answers from all available AIs",
                               use_container_width=True)
    
    # Process queries
    if prompt:
        if single_query:
            with st.spinner("🤔 Thinking..."):
                response = router.query_best_for_use_case(prompt, selected_use_case)
                evaluation = ResponseEvaluator.evaluate_response(response)
                
                st.markdown("---")
                st.subheader("📝 Answer")
                display_response("Best Model", response, evaluation)
        
        elif compare_all:
            with st.spinner("🤔 Asking all AI models..."):
                responses = router.query_all(prompt)
                comparison = ResponseEvaluator.compare_responses(responses)
                
                st.markdown("---")
                st.subheader("📊 Comparison Results")
                
                # Summary
                if comparison.get("successful_models", 0) > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Models Tested", comparison["total_models"])
                    with col2:
                        st.metric("Successful", comparison["successful_models"])
                    with col3:
                        if "fastest_model" in comparison:
                            st.metric("Fastest", comparison["fastest_model"],
                                     delta=f"{comparison['fastest_latency']:.2f}s")
                    
                    st.markdown("---")
                    
                    # Individual responses
                    for name, response in responses.items():
                        evaluation = comparison["evaluations"][name]
                        with st.expander(f"🤖 {name.upper()}", expanded=False):
                            display_response(name, response, evaluation)
                else:
                    st.error("❌ All models failed. Please check your API keys and try again.")
    else:
        st.info("👆 Please enter a question above to get started!")


if __name__ == "__main__":
    main()
