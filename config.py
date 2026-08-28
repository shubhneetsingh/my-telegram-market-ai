"""
Configuration loader and settings management for the Multi-Agent Telegram AI Market Assistant.
"""

import os
from dotenv import load_dotenv

# Load local .env file
load_dotenv()

# Telegram Credentials
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()

# AI Provider Settings
AI_PROVIDER = os.getenv("AI_PROVIDER", "nvidia").lower().strip()
AI_MODEL = os.getenv("AI_MODEL", "").strip()

# Per-Agent Specialized Model Overrides (Optional - defaults to AI_MODEL if not specified)
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "").strip()
NEWS_MODEL = os.getenv("NEWS_MODEL", "").strip()
TECHNICAL_MODEL = os.getenv("TECHNICAL_MODEL", "").strip()
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "").strip()
SYNTHESIZER_MODEL = os.getenv("SYNTHESIZER_MODEL", "").strip()

# Provider API Keys
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").strip()

# Memory settings
try:
    MAX_MEMORY_TURNS = int(os.getenv("MAX_MEMORY_TURNS", 16))
except ValueError:
    MAX_MEMORY_TURNS = 16


def get_ai_client_config():
    """
    Returns (base_url, api_key, default_model) based on the configured AI_PROVIDER.
    """
    provider = AI_PROVIDER

    if provider == "nvidia":
        base_url = "https://integrate.api.nvidia.com/v1"
        api_key = NVIDIA_API_KEY
        default_model = "nvidia/llama-3.1-nemotron-70b-instruct"
    elif provider == "groq":
        base_url = "https://api.groq.com/openai/v1"
        api_key = GROQ_API_KEY
        default_model = "llama-3.3-70b-versatile"
    elif provider == "openrouter":
        base_url = "https://openrouter.ai/api/v1"
        api_key = OPENROUTER_API_KEY
        default_model = "deepseek/deepseek-r1"
    elif provider == "deepseek":
        base_url = "https://api.deepseek.com/v1"
        api_key = DEEPSEEK_API_KEY
        default_model = "deepseek-chat"
    elif provider == "ollama":
        base_url = OLLAMA_BASE_URL
        api_key = "ollama"
        default_model = "llama3.2"
    else:
        base_url = "https://integrate.api.nvidia.com/v1"
        api_key = NVIDIA_API_KEY
        default_model = "nvidia/llama-3.1-nemotron-70b-instruct"

    model = AI_MODEL if AI_MODEL else default_model
    return base_url, api_key, model


def resolve_agent_model(agent_role: str) -> str:
    """Resolves the best model for a specific agent role."""
    base_url, api_key, default_model = get_ai_client_config()

    if agent_role == "router":
        return ROUTER_MODEL or default_model
    elif agent_role == "news":
        return NEWS_MODEL or default_model
    elif agent_role == "technical":
        return TECHNICAL_MODEL or default_model
    elif agent_role == "critic":
        return CRITIC_MODEL or default_model
    elif agent_role == "synthesizer":
        return SYNTHESIZER_MODEL or default_model
# AI Persona & System Prompt
SYSTEM_PROMPT = """You are your personal Multi-Agent AI Market Intelligence & Quantitative Trading Partner.

COMMUNICATION & CONVERSATIONAL STYLE:
1. Short, Punchy & Direct: NEVER output giant walls of text, 10-point essays, or academic fluff. Traders need quick, actionable clarity. Keep responses concise (3-4 bullet points max or 2 short paragraphs).
2. Natural & Engaging: Talk like a sharp, institutional trading partner. Be conversational, crisp, and direct.
3. Interactive Follow-up: Always wrap up with a quick, natural question or offer (e.g. "What ticker are you watching right now?", "Want me to run the 1H levels on Gold?").
4. Grounded in Evidence: Use the provided multi-timeframe numbers, EMAs, RSI, and news cleanly without inventing chart facts.
5. Identity Rule: Do NOT refer to yourself as Nexsee, Nexy, or IMX. You are a standalone personal market intelligence assistant.
"""
