# 🤖 Multi-LLM System

**Tech for Social Good - Democratizing AI for Everyone**

A simple, accessible multi-LLM system designed to help users (especially elderly, disabled, and non-technical people) find and use the best AI model for their needs without technical expertise.

## 🌟 Mission

Enable users who face barriers to easily evaluate, find, and select the optimal AI model for their needs in healthcare, social welfare, accessibility, and everyday assistance.

## ✨ Features

- **🎯 Multiple Free LLM Providers**: HuggingFace, Google Gemini, Groq, Ollama, OpenRouter
- **🧭 Smart Routing**: Automatically select the best model for your use case
- **📊 Easy Comparison**: Compare responses from all models side-by-side
- **♿ Accessibility First**: Large fonts, high contrast, simple navigation
- **📖 Readability Scoring**: Understand how easy responses are to read
- **💰 Cost-Free Focus**: Prioritizes free-tier services and local models
- **🌐 Web Interface**: Beautiful, simple Streamlit interface

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/multi-llm-system.git
cd multi-llm-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys (get them from the links below):

- **HuggingFace**: [Get API Key](https://huggingface.co/settings/tokens)
- **Google Gemini**: [Get API Key](https://makersuite.google.com/app/apikey)
- **Groq**: [Get API Key](https://console.groq.com/keys)
- **OpenRouter**: [Get API Key](https://openrouter.ai/keys)
- **Ollama**: [Download](https://ollama.ai/) (runs locally, no API key needed)

> **Note**: You only need at least ONE API key to get started. The system works with any combination of providers!

### 4. Run the Web Interface

```bash
streamlit run app.py
```

The interface will open in your browser at `http://localhost:8501`

### 5. (Optional) Try the Example Script

```bash
python examples/basic_usage.py
```

## 🎨 Web Interface

The Streamlit interface provides:

1. **Simple 3-Step Process**:
   - Choose your need (Healthcare, Accessibility, General, Cost-Effective)
   - Type your question
   - Get your answer or compare all models

2. **Accessibility Features**:
   - Large, readable text (1.3x larger than standard)
   - High contrast colors
   - Clear button labels
   - Simple navigation
   - Readability scoring

3. **Smart Features**:
   - Use case-based model recommendations
   - Response speed ratings
   - Side-by-side comparisons
   - Cost transparency

## 📚 Use Case Examples

### Healthcare Information
```python
from src.router import MultiLLMRouter, UseCase
from src.providers import GeminiProvider

router = MultiLLMRouter()
router.register_provider("gemini", GeminiProvider(api_key="your-key"))

response = router.query_best_for_use_case(
    "What are the warning signs of a heart attack?",
    UseCase.HEALTHCARE
)
print(response.content)
```

### Accessibility Support
```python
response = router.query_best_for_use_case(
    "How can I make my smartphone easier to use with limited vision?",
    UseCase.ACCESSIBILITY
)
print(response.content)
```

### Compare All Models
```python
responses = router.query_all("What is diabetes?")
for name, response in responses.items():
    print(f"{name}: {response.content[:100]}...")
```

## 🔧 Supported Models

### HuggingFace Inference API (Free Tier)
- Mistral-7B-Instruct
- Llama-2-7b-chat
- Zephyr-7b-beta

### Google Gemini (Free Tier)
- gemini-pro

### Groq (Free Tier - Very Fast!)
- llama3-8b-8192
- mixtral-8x7b-32768
- gemma-7b-it

### Ollama (Local - Completely Free)
- llama3
- mistral
- phi3

### OpenRouter (Free Models)
- meta-llama/llama-3-8b-instruct:free
- google/gemma-7b-it:free
- mistralai/mistral-7b-instruct:free

## 🏗️ Project Structure

```
multi-llm-system/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example             # Example environment variables
├── .gitignore               # Git ignore file
├── app.py                   # Streamlit web interface
├── src/
│   ├── __init__.py
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract base provider
│   │   ├── huggingface.py   # HuggingFace provider
│   │   ├── gemini.py        # Google Gemini provider
│   │   ├── groq.py          # Groq provider
│   │   ├── ollama.py        # Ollama local provider
│   │   └── openrouter.py    # OpenRouter provider
│   ├── router.py            # Multi-LLM router
│   └── evaluator.py         # Response evaluation
└── examples/
    └── basic_usage.py       # Example usage script
```

## 💡 How It Works

### 1. Provider Abstraction
All LLM providers implement a common interface (`BaseLLMProvider`) that returns standardized responses (`LLMResponse`) containing:
- Model name
- Generated content
- Latency (response time)
- Token usage
- Estimated cost

### 2. Smart Routing
The `MultiLLMRouter` intelligently selects models based on use cases:
- **Healthcare**: Prefers accurate, reliable models (Gemini, Groq)
- **Accessibility**: Prefers fast, clear models (Groq, Gemini)
- **General**: Balances speed and quality (Groq, Ollama)
- **Cost-Sensitive**: Prefers free/local models (Ollama, Groq)

### 3. Response Evaluation
The `ResponseEvaluator` provides:
- Readability scoring (Flesch Reading Ease)
- Speed ratings
- Comparison metrics
- Quality assessments

## 🔒 Privacy & Security

- **Local-First Option**: Use Ollama to run models entirely on your computer
- **No Data Storage**: Queries are sent directly to providers, not stored
- **API Key Security**: Keys stored in `.env` file (never committed to git)
- **Transparent**: See exactly which model responds to your query

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Add New Providers**: Implement more free LLM services
2. **Improve Accessibility**: Enhance UI/UX for diverse users
3. **Better Routing**: Improve use case detection and model selection
4. **Documentation**: Help others understand and use the system
5. **Bug Fixes**: Report and fix issues

### Development Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/multi-llm-system.git
cd multi-llm-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest

# Run the app
streamlit run app.py
```

## 📋 Requirements

- Python 3.8+
- Internet connection (except for Ollama)
- At least one API key (or Ollama installed locally)

## 🐛 Troubleshooting

### "No providers available"
- Check that you have at least one API key in your `.env` file
- Make sure the `.env` file is in the root directory

### Ollama connection error
- Install Ollama from https://ollama.ai/
- Start Ollama: `ollama serve`
- Pull a model: `ollama pull llama3`

### API rate limits
- Free tiers have usage limits
- Try using Ollama for unlimited local inference
- Spread requests across multiple providers

### Slow responses
- Some models are slower than others
- Groq is known for very fast inference
- Local Ollama speed depends on your hardware

## 📄 License

MIT License - feel free to use this project for any purpose!

## 🙏 Acknowledgments

- **HuggingFace** for free model hosting
- **Google** for Gemini API
- **Groq** for lightning-fast inference
- **Ollama** for local model support
- **OpenRouter** for multi-model access
- The open-source community for making AI accessible

## 📞 Support

- 📧 Email: support@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/multi-llm-system/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/multi-llm-system/discussions)

---

**Made with ❤️ for accessible AI**
