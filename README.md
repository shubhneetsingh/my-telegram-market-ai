# 📈 Trade with Bebo — Multi-Agent AI Market Assistant

**Trade with Bebo** is an institutional-grade personal Telegram AI trading and market intelligence assistant powered by **Bebo**, a Multi-Agent AI System with **Deterministic Multi-Timeframe OHLCV Technicals** and **Adversarial Risk Governance**.

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

## 🌐 24/7 Cloud Deployment (Render + UptimeRobot)

### Step 1: Push to GitHub
Create a new private or public repository on [github.com](https://github.com/new), then run in this folder:
```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Render
1. Go to **[dashboard.render.com](https://dashboard.render.com)** and click **New +** ➔ **Web Service**.
2. Connect your GitHub repository.
3. Settings:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py`
   - **Plan**: `Free`
4. Add **Environment Variables** under the "Environment" tab:
   - `TELEGRAM_TOKEN` = `your_bot_token`
   - `NVIDIA_API_KEY` = `nvapi-your_key`
   - `AI_PROVIDER` = `nvidia`
   - `AI_MODEL` = `meta/llama-3.2-11b-vision-instruct`
5. Click **Create Web Service**.

### Step 3: Keep-Alive with UptimeRobot (24/7 Free)
1. Go to **[uptimerobot.com](https://uptimerobot.com)** (free account).
2. Click **Add New Monitor**:
   - **Monitor Type**: `HTTP(s)`
   - **Friendly Name**: `My Telegram Market AI`
   - **URL (or IP)**: `https://your-bot-name.onrender.com/health`
   - **Monitoring Interval**: `5 minutes`
3. Click **Create Monitor**.

*UptimeRobot will ping your server every 5 minutes, preventing Render from sleeping and keeping your Telegram bot 100% online 24/7 without needing your PC on!*

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
