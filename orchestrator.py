"""
True Multi-Agent Orchestrator with Robust Failure Handling, Graceful Degradation,
Adversarial Critic Overrides, and Deterministic Evidence Confluence Scoring.
"""

import asyncio
import json
import logging
import re
import time
from typing import Dict, Any, List, Optional
from ai_engine import ai_engine
from config import resolve_agent_model
from market_data import (
    get_multi_timeframe_technical_data,
    format_ticker_summary,
    search_live_market_news,
    extract_potential_tickers,
)
from risk_engine import calculate_position_and_risk, format_risk_report

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# DETERMINISTIC CONFLUENCE & EVIDENCE SCORER
# ----------------------------------------------------------------------
def calculate_evidence_confluence_score(
    data: Dict[str, Any],
    news_sentiment: str = "NEUTRAL",
    news_status: str = "VERIFIED",
) -> Dict[str, Any]:
    """
    Computes an objective mathematical evidence quality & confluence score (0-100).
    Explicitly represents EVIDENCE CONFLUENCE, not win probability.
    """
    score = 0
    breakdown = []

    daily = data.get("daily_technical", {})
    hourly = data.get("hourly_technical", {})

    # 1. Multi-Timeframe EMA Alignment (Max 35 pts)
    d_struct = daily.get("structure", "")
    h_struct = hourly.get("structure", "")

    if "STRONG BULLISH" in d_struct and "BULLISH" in h_struct:
        score += 35
        breakdown.append("✅ 1D & 1H Strong Bullish EMA Confluence (+35 pts)")
    elif "STRONG BEARISH" in d_struct and "BEARISH" in h_struct:
        score += 35
        breakdown.append("✅ 1D & 1H Strong Bearish EMA Confluence (+35 pts)")
    elif "BULLISH" in d_struct or "BEARISH" in d_struct:
        score += 20
        breakdown.append("⚡ Partial Multi-Timeframe Trend Alignment (+20 pts)")
    else:
        score += 10
        breakdown.append("⚠️ Choppy / Range-Bound Market Structure (+10 pts)")

    # 2. RSI Momentum & Overextension (Max 25 pts)
    rsi_1d = daily.get("rsi_14", 50)
    if 40 <= rsi_1d <= 65:
        score += 25
        breakdown.append(f"✅ Daily RSI Balanced at {rsi_1d} (No Overextension) (+25 pts)")
    elif rsi_1d > 75:
        score += 5
        breakdown.append(f"🔴 Daily RSI Extreme Overbought at {rsi_1d} (+5 pts)")
    elif rsi_1d < 25:
        score += 5
        breakdown.append(f"🟢 Daily RSI Extreme Oversold at {rsi_1d} (+5 pts)")
    else:
        score += 15
        breakdown.append(f"⚡ Daily RSI Moderate at {rsi_1d} (+15 pts)")

    # 3. Volatility / ATR Grounding (Max 20 pts)
    atr = daily.get("atr_14", 0)
    if atr > 0:
        score += 20
        breakdown.append(f"✅ Volatility Grounded (Daily ATR: ${atr:,.2f}) (+20 pts)")
    else:
        score += 5
        breakdown.append("⚠️ ATR Data Missing (+5 pts)")

    # 4. News & Macro Confluence (Max 20 pts)
    if news_status == "UNVERIFIED":
        score += 0
        breakdown.append("⚠️ News Data Unverified / Missing (0 pts)")
    elif "BULLISH" in news_sentiment.upper() or "BEARISH" in news_sentiment.upper():
        score += 20
        breakdown.append(f"✅ Macro Catalyst Aligned ({news_sentiment.upper()}) (+20 pts)")
    else:
        score += 10
        breakdown.append("⚡ Neutral Macro Backdrop (+10 pts)")

    # Qualitative Rating
    if score >= 80:
        rating = "🟢 HIGH CONFLUENCE"
    elif score >= 60:
        rating = "🟡 MODERATE CONFLUENCE"
    else:
        rating = "🔴 LOW CONFLUENCE / HIGH RISK"

    return {
        "score": score,
        "rating": rating,
        "breakdown": breakdown,
    }


# ----------------------------------------------------------------------
# AGENT SYSTEM PROMPTS (STRUCTURED JSON)
# ----------------------------------------------------------------------
ROUTER_PROMPT = """You are Agent 0: The Intent Router.
Classify the user's intent into EXACTLY ONE category:
- TECHNICAL_ANALYSIS
- TRADE_PLAN
- NEWS_MACRO
- PRICE_CHECK
- RISK_CALC
- GENERAL_CHAT

Output ONLY the category name. Nothing else."""

NEWS_AGENT_PROMPT = """You are Agent 1: The Macro & News Intelligence Specialist.
Analyze the provided live web search results.
Extract fundamental drivers and output a STRICT JSON object:
{
  "catalyst": "Brief 1-2 sentence explanation of the primary news driver",
  "sentiment": "BULLISH | BEARISH | NEUTRAL | UNCERTAIN",
  "affected_assets": ["List of assets"],
  "market_risk": "LOW | MEDIUM | HIGH",
  "uncertainties": "Any missing or conflicting news factors"
}
Output ONLY valid JSON."""

TECHNICAL_AGENT_PROMPT = """You are Agent 2: The Quantitative Technical Analyst.
You are given verified 1D and 1H OHLCV indicators (EMAs 20/50/200, RSI 14, ATR 14, 20-period Swings).
Formulate a disciplined technical thesis and output a STRICT JSON object:
{
  "market_regime": "TRENDING_BULLISH | TRENDING_BEARISH | RANGING | BREAKOUT",
  "trend_1d": "BULLISH | BEARISH | NEUTRAL",
  "trend_1h": "BULLISH | BEARISH | NEUTRAL",
  "key_support": "Price level or zone",
  "key_resistance": "Price level or zone",
  "momentum_status": "RSI assessment",
  "thesis": "Concise 2-3 sentence technical justification based strictly on the provided numbers",
  "invalidation_level": "Exact price level where this thesis becomes completely invalid"
}
Output ONLY valid JSON."""

CRITIC_AGENT_PROMPT = """You are Agent 3: The Adversarial Risk & Critic Verifier.
You are an independent validator from a separate model family. Your role is to find vulnerabilities in the Technical Thesis and News Context using the raw quantitative data.

Audit Rules:
1. Trend Contradiction: Is the thesis fighting higher timeframe (1D/1H) EMAs?
2. Overextension: Is RSI > 75 (buying top) or < 25 (selling bottom)?
3. Invalidation & Volatility: Is the invalidation level placed sensibly relative to ATR and recent swings?
4. Macro Conflicts: Are there high-impact news catalysts that could invalidate technicals?

Output a STRICT JSON object:
{
  "verdict": "APPROVED | CAUTION | INVALIDATED",
  "critique_points": [
    "Point 1: Specific vulnerability or confirmation",
    "Point 2: Specific vulnerability or confirmation"
  ],
  "hard_rule_violations": "None or specific violations",
  "action_guidance": "Clear, direct advice for the user"
}
Output ONLY valid JSON."""

SYNTHESIZER_PROMPT = """You are Agent 4: The Executive Synthesizer.
Combine the quantitative market data, technical analysis, news catalyst, critic verification, and evidence score into an executive, highly readable Telegram markdown briefing.

STRICT OVERRIDE GOVERNANCE RULES:
- If Critic Verdict is 'INVALIDATED': You MUST display '⛔ TRADE SETUP INVALIDATED: High Risk / Contradictory Conditions' at the top. Do NOT recommend entering.
- If Critic Verdict is 'CAUTION': Emphasize the exact waiting condition or confirmation needed before entering.
- If Critic Verdict is 'APPROVED': Present the high-confluence trade framework with the invalidation level clearly marked.
- If Critic is 'UNAVAILABLE': Default to '⚠️ CRITIC REVIEW UNAVAILABLE: Proceed with caution'.

Format with clean bold headers, emojis, and structured bullet points."""


# ----------------------------------------------------------------------
# JSON SAFE PARSER HELPER
# ----------------------------------------------------------------------
def parse_json_safely(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extracts and parses JSON object from model output."""
    if not raw_text:
        return None
    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(raw_text)
    except Exception:
        return None


# ----------------------------------------------------------------------
# ORCHESTRATOR CLASS
# ----------------------------------------------------------------------
class MultiAgentOrchestrator:
    def __init__(self):
        self.ai = ai_engine

    async def route_intent(self, user_message: str) -> str:
        """Fast heuristic + model intent router."""
        lower = user_message.lower().strip()

        # 1. Fast Conversational / Capability / Greeting Detection
        capabilities = [
            "what can you do", "what you can do", "what do you do", "who are you", 
            "how can you help", "tell me about yourself", "what are your features",
            "hello", "hi", "hey", "good morning", "good evening", "how are you", "help"
        ]
        if any(phrase in lower for phrase in capabilities):
            return "GENERAL_CHAT"

        # 2. Fast Command / Keyword Detection
        if lower.startswith("/price"):
            return "PRICE_CHECK"
        if lower.startswith("/news"):
            return "NEWS_MACRO"
        if lower.startswith("/analyze"):
            return "TRADE_PLAN"
        if any(w in lower for w in ["buy or sell", "should i buy", "should i sell", "enter long", "enter short", "trade setup", "trade plan"]):
            return "TRADE_PLAN"
        if any(w in lower for w in ["chart", "trend", "structure", "technical", "rsi", "ema", "support", "resistance"]):
            return "TECHNICAL_ANALYSIS"
        if any(w in lower for w in ["risk", "lot size", "position size", "stop loss distance"]):
            return "RISK_CALC"
        if any(w in lower for w in ["fed", "fomc", "cpi", "inflation", "war", "rate cut", "earnings"]):
            return "NEWS_MACRO"

        try:
            model = resolve_agent_model("router")
            response = await self.ai.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": ROUTER_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=30,
            )
            intent = response.choices[0].message.content.strip().upper()
            return intent if intent in ["TECHNICAL_ANALYSIS", "TRADE_PLAN", "NEWS_MACRO", "PRICE_CHECK", "RISK_CALC", "GENERAL_CHAT"] else "GENERAL_CHAT"
        except Exception:
            return "GENERAL_CHAT"

    async def run_full_pipeline(
        self,
        user_message: str,
        chat_history: List[Dict[str, str]],
    ) -> str:
        """
        Full Multi-Agent Pipeline with Failure Handling & Graceful Degradation:
        1. Agent 0: Router
        2. Quant OHLCV Engine + DuckDuckGo Live Search
        3. Parallel: Agent 1 (News) + Agent 2 (Technicals)
        4. Deterministic Confluence & Evidence Scorer
        5. Agent 3: Adversarial Critic
        6. Agent 4: Executive Synthesizer (with Hard Override & Fallback)
        """
        intent = await self.route_intent(user_message)
        logger.info(f"Orchestrator Intent: {intent} for '{user_message}'")

        tickers = extract_potential_tickers(user_message)
        primary_symbol = tickers[0] if tickers else None

        # -------------------------------------------------------------
        # BRANCH 1: SIMPLE PRICE CHECK
        # -------------------------------------------------------------
        if intent == "PRICE_CHECK":
            if not primary_symbol:
                return "⚠️ Please specify a symbol for price check (e.g. `/price NVDA`, `/price BTC`, `/price GOLD`)."
            data = await get_multi_timeframe_technical_data(primary_symbol)
            if data:
                return format_ticker_summary(data)
            return f"❌ Could not retrieve market data for `{primary_symbol}`."

        # -------------------------------------------------------------
        # BRANCH 2: NEWS & MACRO FOCUS
        # -------------------------------------------------------------
        if intent == "NEWS_MACRO":
            # Only do DuckDuckGo search if there is an explicit symbol or news query
            search_query = primary_symbol or user_message.replace("/news", "").strip()
            if not search_query:
                return "⚠️ Please specify a topic or asset for news search (e.g. `/news Fed rate decision` or `/news Gold`)."

            news_raw = await search_live_market_news(f"{search_query} financial market news today")
            if not news_raw:
                return f"📰 **Macro & News Intelligence:** No recent breaking headlines found for `{search_query}`."

            news_analysis = await self._run_agent(
                NEWS_AGENT_PROMPT,
                f"User Query: {user_message}\n\nSearch Context:\n{news_raw}",
                role="news",
            )
            return f"📰 **Macro & News Intelligence:**\n\n{news_analysis}"

        # -------------------------------------------------------------
        # BRANCH 3: FULL MULTI-AGENT TRADE PLAN & CRITIC PIPELINE
        # -------------------------------------------------------------
        if primary_symbol or intent in ["TECHNICAL_ANALYSIS", "TRADE_PLAN"]:
            if not primary_symbol:
                # If user asked for analysis without mentioning a symbol
                return "📊 Which asset would you like me to analyze? (e.g., `/analyze NVDA`, `/analyze BTC`, or ask *\"Should I enter Gold long?\"*)"

            symbol = primary_symbol
            data = await get_multi_timeframe_technical_data(symbol)

            # Technical Data Failure Handling
            if not data:
                logger.warning(f"Technical data fetch failed for {symbol}")
                return f"⚠️ **Data Insufficient**: Unable to fetch live multi-timeframe data for `{symbol}`. Trade plan cannot be generated without verified data."

            summary_block = format_ticker_summary(data)

            # Live Search Fetch
            news_raw = await search_live_market_news(f"{data['raw_symbol']} market news today", max_results=3)

            # PARALLEL AGENT EXECUTION: Agent 1 (News) & Agent 2 (Technical)
            news_task = self._run_agent(
                NEWS_AGENT_PROMPT,
                f"Asset: {data['raw_symbol']}\nNews:\n{news_raw or 'No breaking news.'}",
                role="news",
            )
            technical_task = self._run_agent(
                TECHNICAL_AGENT_PROMPT,
                f"Asset: {data['raw_symbol']}\n\nQuantitative Evidence:\n{summary_block}\n\nUser Question:\n{user_message}",
                role="technical",
            )

            news_raw_out, tech_raw_out = await asyncio.gather(news_task, technical_task, return_exceptions=True)

            # ---------------------------------------------------------
            # FAILURE HANDLING: AGENT 1 (NEWS)
            # ---------------------------------------------------------
            news_dict = None
            news_status = "VERIFIED"
            if isinstance(news_raw_out, Exception) or not news_raw_out:
                news_status = "UNVERIFIED"
                news_dict = {"catalyst": "News extraction unavailable", "sentiment": "NEUTRAL", "market_risk": "MEDIUM"}
            else:
                news_dict = parse_json_safely(str(news_raw_out))
                if not news_dict:
                    news_status = "UNVERIFIED"
                    news_dict = {"catalyst": str(news_raw_out), "sentiment": "NEUTRAL", "market_risk": "MEDIUM"}

            # ---------------------------------------------------------
            # FAILURE HANDLING: AGENT 2 (TECHNICAL)
            # ---------------------------------------------------------
            tech_dict = None
            if isinstance(tech_raw_out, Exception) or not tech_raw_out:
                return (
                    f"⚠️ **Technical Analysis Error**: Technical Agent failed to generate a thesis.\n\n"
                    f"**Ground-Truth Quant Data Still Available:**\n{summary_block}"
                )
            tech_dict = parse_json_safely(str(tech_raw_out))
            tech_payload = json.dumps(tech_dict, indent=2) if tech_dict else str(tech_raw_out)

            # ---------------------------------------------------------
            # DETERMINISTIC EVIDENCE & CONFLUENCE SCORING
            # ---------------------------------------------------------
            sentiment = news_dict.get("sentiment", "NEUTRAL") if news_dict else "NEUTRAL"
            evidence = calculate_evidence_confluence_score(data, sentiment, news_status)
            evidence_summary = (
                f"📊 **Evidence Confluence Score**: `{evidence['score']}/100` — {evidence['rating']}\n"
                + "\n".join(evidence["breakdown"])
            )

            # ---------------------------------------------------------
            # AGENT 3: ADVERSARIAL CRITIC & VERIFICATION
            # ---------------------------------------------------------
            critic_payload = (
                f"Asset: {data['raw_symbol']}\n\n"
                f"Quantitative Evidence:\n{summary_block}\n\n"
                f"Technical Thesis:\n{tech_payload}\n\n"
                f"News Context ({news_status}):\n{json.dumps(news_dict, indent=2)}\n\n"
                f"Evidence Confluence:\n{evidence_summary}"
            )

            critic_raw_out = await self._run_agent(CRITIC_AGENT_PROMPT, critic_payload, role="critic")

            # ---------------------------------------------------------
            # FAILURE HANDLING: AGENT 3 (CRITIC)
            # RULE: NEVER AUTO-APPROVE IF CRITIC FAILS!
            # ---------------------------------------------------------
            critic_dict = parse_json_safely(critic_raw_out)
            if not critic_dict:
                critic_dict = {
                    "verdict": "CAUTION",
                    "critique_points": ["⚠️ Adversarial Critic returned non-standard format; defaulting to CAUTION."],
                    "action_guidance": "Proceed with caution and await manual confirmation.",
                }

            # ---------------------------------------------------------
            # AGENT 4: EXECUTIVE SYNTHESIZER
            # ---------------------------------------------------------
            final_payload = (
                f"PRODUCE THE FINAL TELEGRAM BRIEFING FOR THE USER:\n\n"
                f"### 1. LIVE QUANT EVIDENCE:\n{summary_block}\n\n"
                f"### 2. EVIDENCE CONFLUENCE SCORE:\n{evidence_summary}\n\n"
                f"### 3. TECHNICAL ANALYSIS (Agent 2):\n{tech_payload}\n\n"
                f"### 4. MACRO CONTEXT (Agent 1 - {news_status}):\n{json.dumps(news_dict, indent=2)}\n\n"
                f"### 5. ADVERSARIAL CRITIC VERDICT (Agent 3):\n{json.dumps(critic_dict, indent=2)}\n\n"
                f"### USER'S ORIGINAL QUESTION:\n{user_message}"
            )

            try:
                final_briefing = await self._run_agent(SYNTHESIZER_PROMPT, final_payload, role="synthesizer")
                return final_briefing
            except Exception:
                # Deterministic Fallback if Synthesizer fails
                verdict = critic_dict.get("verdict", "CAUTION")
                verdict_emoji = "🟢" if verdict == "APPROVED" else "🟡" if verdict == "CAUTION" else "🔴"
                return (
                    f"### {verdict_emoji} **Market Briefing: {data['raw_symbol']}**\n\n"
                    f"{summary_block}\n\n"
                    f"{evidence_summary}\n\n"
                    f"🛡️ **Critic Verdict**: `{verdict}`\n"
                    f"• {critic_dict.get('action_guidance', 'Awaiting confirmation.')}\n\n"
                    f"🎯 **Technical Thesis**: {tech_dict.get('thesis', 'N/A') if tech_dict else 'N/A'}\n"
                    f"🛑 **Invalidation Level**: `{tech_dict.get('invalidation_level', 'N/A') if tech_dict else 'N/A'}`"
                )

        # -------------------------------------------------------------
        # BRANCH 4: GENERAL CHAT & CONCEPTUAL DISCUSSION
        # -------------------------------------------------------------
        lower_msg = user_message.lower().strip()
        if any(phrase in lower_msg for phrase in ["what can you do", "what you can do", "what do you do", "what are your features", "capabilities", "what are your commands"]):
            return (
                "⚡ **Here is what I do for your trading:**\n\n"
                "• 📊 **Multi-Timeframe Analysis** (`/analyze <symbol>`)\n"
                "  ➔ Live 1D & 1H OHLCV data, EMAs (20/50/200), RSI & key swings.\n\n"
                "• 🛡️ **Adversarial Trade Critic**\n"
                "  ➔ Audits your trade ideas for overextension & trend contradictions.\n\n"
                "• 🎯 **Deterministic Risk Math** (`/risk <entry> <stop> <target>`)\n"
                "  ➔ Exact position sizes, dollar risk, and R:R ratios.\n\n"
                "• 📰 **Live Macro & News** (`/news <topic>`)\n"
                "  ➔ Instant catalyst extraction for Fed, CPI, earnings & global markets.\n\n"
                "💡 *Drop any ticker (e.g. `/analyze GOLD` or `What's the trend on BTC?`) and let's break it down!*"
            )

        return await self.ai.generate_response(chat_history)

    async def _run_agent(self, system_prompt: str, user_content: str, role: str = "synthesizer") -> str:
        """Invokes a specific agent role with its dedicated model configuration."""
        target_model = resolve_agent_model(role)
        response = await self.ai.client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=900,
        )
        raw = response.choices[0].message.content or ""
        if "<think>" in raw and "</think>" in raw:
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return raw.strip()


# Global singleton instance
orchestrator = MultiAgentOrchestrator()
