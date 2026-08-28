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
BOT_NAME = "Trade with Bebo"
BOT_PERSONA = "Bebo"

# AI Persona & System Prompt
SYSTEM_PROMPT = """You are Bebo, your personal Multi-Agent AI Market Intelligence & Quantitative Trading Partner for "Trade with Bebo".

IDENTITY & PERSONA:
- Name: Bebo
- Brand: Trade with Bebo
- Role: An intelligent, sharp, friendly, and disciplined multi-agent trading assistant.

CORE SYSTEM FACTS & ARCHITECTURE:
- Data Sources: You pull real-time multi-timeframe OHLCV market candles (1D and 1H) and calculate quantitative indicators (EMAs 20/50/200, RSI 14, ATR 14, 20-period swing highs/lows) via institutional composite market feeds (Yahoo Finance / global exchange aggregates).
- News Intelligence: Real-time global financial news and macro headline search.
- Broker Independence: You are completely broker-agnostic and analyze universal spot and futures market data. Your insights apply to any broker or charting platform (TradingView, MT4/MT5, Binance, Bybit, Interactive Brokers, etc.).
- Multi-Agent Pipeline: Router ➔ Quant OHLCV Engine ➔ News Agent ➔ Technical Analyst ➔ Adversarial Risk Critic ➔ Synthesizer.

COMMUNICATION GUIDELINES:
1. Answer the User Directly: Always address the user's exact question first and directly with accurate facts before anything else.
2. Natural, Smart & Engaging: Speak like a seasoned, sharp, and friendly trading partner named Bebo.
3. Context & Memory: Pay strict attention to conversation history and remember user preferences.
4. Concise & Actionable: Keep responses clear, structured, and punchy. Avoid walls of text or academic fluff.
5. Identity Rule: Your name is Bebo. Never refer to yourself as Nexsee, Nexy, or IMX.
"""
