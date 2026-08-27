# 📈 Multi-Agent Telegram AI Market Assistant

A personal, institutional-grade Telegram AI market assistant powered by a **Multi-Agent Orchestrator**, **Deterministic Multi-Timeframe OHLCV Technicals**, and **Open-Source AI Models from NVIDIA NIM** (DeepSeek-R1, Nemotron 70B, Llama 3.3).

---

## 🏗️ Multi-Agent Architecture

```text
User Question / Trade Setup
            │
            ▼
┌───────────────────────────────────────┐
│     Agent 0: Intent Router            │  (Heuristics + Fast Model)
└──────────────────┬────────────────────┘
                   │
    ┌──────────────┴──────────────┐
    ▼                             ▼
┌──────────────────────┐  ┌──────────────────────┐
│  Quant OHLCV Engine  │  │  Live News Engine    │
│  (1D/1H EMAs, RSI,   │  │  (DuckDuckGo Search) │
│   ATR, Swings)       │  │                      │
└──────────┬───────────┘  └──────────┬───────────┘
           │                         │
           ▼                         ▼
┌──────────────────────┐  ┌──────────────────────┐
│  Agent 2: Technical  │  │  Agent 1: Macro/News │
│  Analyst Agent       │  │  Intelligence Agent  │
└──────────┬───────────┘  └──────────┬───────────┘
           │                         │
           └───────────┬─────────────┘
                       │
                       ▼
         ┌───────────────────────────┐
         │ Agent 3: Critic Verifier  │  (Adversarial Risk Audit)
         └─────────────┬─────────────┘
                       │
                       ▼
         ┌───────────────────────────┐
         │ Deterministic Risk Engine │  (Exact Position Size & R:R)
         └─────────────┬─────────────┘
                       │
                       ▼
         ┌───────────────────────────┐
         │ Agent 4: Synthesizer      │  (Executive Telegram Briefing)
         └───────────────────────────┘
```

---

## ✨ Key Capabilities

1. **Deterministic Technical Engine (`market_data.py`)**:
   - Computes **Daily (1D) and Hourly (1H) EMAs (20, 50, 200)** for structural trend alignment.
   - Computes **RSI (14)** for momentum and overextension checks.
   - Computes **ATR (14)** for volatility-grounded stop loss buffers.
   - Calculates **20-Period Swing Highs & Lows** for real horizontal support & resistance.

2. **Deterministic Risk & Position Sizing (`risk_engine.py`)**:
   - Computes exact unit position size based on account balance and risk %.
   - Calculates exact Risk-to-Reward ratio ($R:R$) and dollar upside/downside.
   - Warns if the stop-loss is placed too tight relative to ATR ($<0.8\times \text{ATR}$).

3. **Adversarial Critic & Verification Agent (`orchestrator.py`)**:
   - Actively audits trade ideas to catch trend contradictions, overbought RSI, or high-impact macro conflicts before issuing a verdict: `[APPROVED | CAUTION | INVALIDATED]`.

4. **Multi-Model / Provider Flexibility**:
   - Plug in **NVIDIA NIM** (`deepseek-ai/deepseek-r1`, `nvidia/llama-3.1-nemotron-70b-instruct`), **Groq**, **OpenRouter**, **DeepSeek**, or local **Ollama** simply by editing `.env`.

---

## 🚀 Quick Setup

### Step 1: Add Credentials to [`.env`](file:///c:/Users/shubh/Downloads/Telegram%20Bot/.env)

```env
TELEGRAM_TOKEN=your_token_from_botfather

AI_PROVIDER=nvidia
AI_MODEL=nvidia/llama-3.1-nemotron-70b-instruct

NVIDIA_API_KEY=nvapi-your_nvidia_api_key_here
```

### Step 2: Start the Bot

```bash
python bot.py
```

---

## 📱 Bot Commands

| Command | Description | Example |
| :--- | :--- | :--- |
| `/analyze <symbol>` | Full multi-agent trade thesis, OHLCV breakdown, and critic review | `/analyze NVDA` or `/analyze BTC` |
| `/price <symbol>` | Instant deterministic price & 1D/1H technical metrics | `/price GOLD` or `/price EURUSD` |
| `/risk <entry> <stop> <tp>` | Deterministic position sizing and R:R calculator | `/risk 100 95 115 10000 1.0` |
| `/news <query>` | Macro news catalyst & sentiment synthesis | `/news Fed interest rates` |
| `/model` | View active AI model and pipeline status | `/model` |
| `/clear` | Clear conversation memory | `/clear` |
| `/help` | Complete help guide | `/help` |
