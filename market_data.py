"""
Market Data & Quantitative Technical Engine.
Fetches real-time quotes, multi-timeframe OHLCV history, and calculates deterministic indicators.
"""

import asyncio
import logging
import re
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import yfinance as yf

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

# Symbol alias mapping for fast user resolution
SYMBOL_MAP = {
    "BTC": "BTC-USD",
    "BITCOIN": "BTC-USD",
    "ETH": "ETH-USD",
    "ETHEREUM": "ETH-USD",
    "SOL": "SOL-USD",
    "SOLANA": "SOL-USD",
    "XRP": "XRP-USD",
    "GOLD": "GC=F",
    "XAUUSD": "GC=F",
    "SILVER": "SI=F",
    "XAGUSD": "SI=F",
    "OIL": "CL=F",
    "CRUDE": "CL=F",
    "BRENT": "BZ=F",
    "SPX": "^GSPC",
    "SP500": "^GSPC",
    "S&P500": "^GSPC",
    "NASDAQ": "^IXIC",
    "NDX": "^IXIC",
    "DOW": "^DJI",
    "DXY": "DX-Y.NYB",
    "DOLLAR": "DX-Y.NYB",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X",
}


def normalize_symbol(symbol: str) -> str:
    """Resolves common ticker names to standard yfinance tickers."""
    cleaned = symbol.strip().upper().replace("$", "")
    return SYMBOL_MAP.get(cleaned, cleaned)


# ----------------------------------------------------------------------
# DETERMINISTIC TECHNICAL INDICATORS CALCULATOR
# ----------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes deterministic technical indicators on OHLCV DataFrame:
    - EMAs (20, 50, 200)
    - RSI (14)
    - ATR (14)
    - 20-period Swing High / Low (Support / Resistance)
    - Volume Trend (relative to 20-period SMA)
    - Market Structure (Trend bias based on EMA alignment & Swings)
    """
    if len(df) < 20:
        return {}

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # 1. EMAs
    ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1] if len(df) >= 50 else None
    ema_200 = close.ewm(span=200, adjust=False).mean().iloc[-1] if len(df) >= 200 else None

    current_price = close.iloc[-1]

    # 2. RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    current_rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

    # 3. ATR (14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(window=14, min_periods=14).mean()
    current_atr = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else float(tr.mean())

    # 4. Swing Highs & Lows (Support / Resistance zones)
    rolling_high = high.rolling(window=20).max().iloc[-1]
    rolling_low = low.rolling(window=20).min().iloc[-1]

    # 5. Volume Profile
    avg_vol_20 = volume.rolling(window=20).mean().iloc[-1] if "Volume" in df and volume.sum() > 0 else 0
    curr_vol = volume.iloc[-1] if "Volume" in df else 0
    vol_ratio = (curr_vol / avg_vol_20) if avg_vol_20 > 0 else 1.0

    # 6. Deterministic Market Structure Assessment
    structure = "NEUTRAL / RANGING"
    if ema_50 and ema_200:
        if current_price > ema_20 > ema_50 > ema_200:
            structure = "STRONG BULLISH (Full EMA Alignment)"
        elif current_price > ema_50 and current_price > ema_200:
            structure = "BULLISH BIAS (Above 50 & 200 EMA)"
        elif current_price < ema_20 < ema_50 < ema_200:
            structure = "STRONG BEARISH (Full EMA Breakdown)"
        elif current_price < ema_50 and current_price < ema_200:
            structure = "BEARISH BIAS (Below 50 & 200 EMA)"
    elif ema_50:
        if current_price > ema_20 > ema_50:
            structure = "BULLISH BIAS (Above 20 & 50 EMA)"
        elif current_price < ema_20 < ema_50:
            structure = "BEARISH BIAS (Below 20 & 50 EMA)"

    return {
        "current_price": float(current_price),
        "ema_20": float(ema_20),
        "ema_50": float(ema_50) if ema_50 is not None else None,
        "ema_200": float(ema_200) if ema_200 is not None else None,
        "rsi_14": round(current_rsi, 2),
        "atr_14": round(current_atr, 4),
        "swing_high_20": float(rolling_high),
        "swing_low_20": float(rolling_low),
        "volume_ratio_20": round(float(vol_ratio), 2),
        "structure": structure,
    }


# ----------------------------------------------------------------------
# MULTI-TIMEFRAME DATA RETRIEVER
# ----------------------------------------------------------------------
def fetch_multi_timeframe_technical_data_sync(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetches multi-timeframe historical candle data (1H, 1D) and calculates deterministic metrics.
    """
    normalized = normalize_symbol(symbol)
    try:
        ticker = yf.Ticker(normalized)

        # 1. Fetch Daily (1D) for macro trend & key levels (60 days)
        df_daily = ticker.history(period="60d", interval="1d")
        # 2. Fetch Hourly (1H) for swing/intermediate structure (14 days)
        df_hourly = ticker.history(period="14d", interval="1h")

        if df_daily.empty and df_hourly.empty:
            return None

        daily_metrics = compute_indicators(df_daily) if not df_daily.empty else {}
        hourly_metrics = compute_indicators(df_hourly) if not df_hourly.empty else {}

        # Basic quote stats
        fast_info = ticker.fast_info
        last_price = getattr(fast_info, "last_price", None)
        prev_close = getattr(fast_info, "previous_close", None)

        if last_price is None or last_price == 0:
            if not df_daily.empty:
                last_price = float(df_daily["Close"].iloc[-1])
                prev_close = float(df_daily["Close"].iloc[-2]) if len(df_daily) > 1 else float(df_daily["Open"].iloc[-1])

        change = 0.0
        change_pct = 0.0
        if prev_close and prev_close > 0 and last_price:
            change = last_price - prev_close
            change_pct = (change / prev_close) * 100

        currency = getattr(fast_info, "currency", "USD") or "USD"

        return {
            "symbol": normalized,
            "raw_symbol": symbol.upper(),
            "last_price": last_price,
            "change": change,
            "change_pct": change_pct,
            "currency": currency,
            "daily_technical": daily_metrics,
            "hourly_technical": hourly_metrics,
        }
    except Exception as e:
        logger.warning(f"Error fetching technical data for {symbol}: {e}")
        return None


async def get_multi_timeframe_technical_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Async wrapper for technical data fetch."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fetch_multi_timeframe_technical_data_sync, symbol)


# ----------------------------------------------------------------------
# LIVE SPOT QUOTE (FAST SUMMARY)
# ----------------------------------------------------------------------
def fetch_ticker_data_sync(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetches fast price quote and basic stats."""
    return fetch_multi_timeframe_technical_data_sync(symbol)


async def get_live_ticker_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """Async wrapper for live quote."""
    return await get_multi_timeframe_technical_data(symbol)


def format_ticker_summary(data: Dict[str, Any]) -> str:
    """Formats multi-timeframe quantitative data into a clean, markdown block."""
    sign = "+" if data["change"] >= 0 else ""
    trend_emoji = "🟢" if data["change"] >= 0 else "🔴"
    price = data["last_price"]
    price_val = f"{price:,.4f}" if price < 1 else f"{price:,.2f}"

    lines = [
        f"📊 **{data['raw_symbol']} ({data['symbol']})**",
        f"💵 **Price**: ${price_val} {data['currency']}",
        f"{trend_emoji} **24h Change**: {sign}{data['change']:.2f} ({sign}{data['change_pct']:.2f}%)",
    ]

    # Daily structure
    daily = data.get("daily_technical", {})
    if daily:
        lines.append("\n📈 **Daily (1D) Technical Evidence:**")
        lines.append(f"• **Structure**: `{daily.get('structure', 'N/A')}`")
        lines.append(f"• **RSI (14)**: `{daily.get('rsi_14', 'N/A')}`")
        lines.append(f"• **ATR (14)**: `${daily.get('atr_14', 'N/A')}`")
        if daily.get("ema_50") and daily.get("ema_200"):
            lines.append(f"• **Key EMAs**: 20: `${daily['ema_20']:,.2f}` | 50: `${daily['ema_50']:,.2f}` | 200: `${daily['ema_200']:,.2f}`")
        lines.append(f"• **20-Period Range**: `${daily.get('swing_low_20', 0):,.2f}` — `${daily.get('swing_high_20', 0):,.2f}`")

    # Hourly structure
    hourly = data.get("hourly_technical", {})
    if hourly:
        lines.append("\n⏱️ **Hourly (1H) Swing Context:**")
        lines.append(f"• **Structure**: `{hourly.get('structure', 'N/A')}`")
        lines.append(f"• **RSI (14)**: `{hourly.get('rsi_14', 'N/A')}`")
        lines.append(f"• **Key Swing Low/High**: `${hourly.get('swing_low_20', 0):,.2f}` / `${hourly.get('swing_high_20', 0):,.2f}`")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# LIVE WEB SEARCH (DUCKDUCKGO)
# ----------------------------------------------------------------------
def sync_duckduckgo_search(query: str, max_results: int = 2) -> str:
    """Performs fast live web search for financial news and macro events."""
    try:
        with DDGS(timeout=3) as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return ""

            output = ["--- LIVE MARKET HEADLINES ---"]
            for idx, r in enumerate(results, 1):
                title = r.get("title", "").strip()
                body = r.get("body", "").strip()
                output.append(f"[{idx}] {title}\n{body}")
            return "\n\n".join(output)
    except Exception as e:
        logger.warning(f"DuckDuckGo search skipped/timed out: {e}")
        return ""


async def search_live_market_news(query: str, max_results: int = 2) -> str:
    """Asynchronous wrapper for DuckDuckGo news search with fast timeout."""
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(loop.run_in_executor(None, sync_duckduckgo_search, query, max_results), timeout=3.5)
    except asyncio.TimeoutError:
        logger.warning("Live search timed out (exceeded 3.5s limit). Continuing without search context.")
        return ""


def extract_potential_tickers(text: str) -> List[str]:
    """Detects stock, crypto, or forex tickers mentioned in text."""
    cashtags = re.findall(r"\$([A-Za-z]{1,6})", text)
    if cashtags:
        return [c.upper() for c in cashtags]

    words = [w.strip("?,.!:;()").upper() for w in text.split()]
    found = []
    for w in words:
        if w in SYMBOL_MAP:
            found.append(w)
        elif len(w) in [2, 3, 4, 5] and w.isalpha() and w in ["NVDA", "AAPL", "TSLA", "MSFT", "AMD", "META", "AMZN", "GOOGL"]:
            found.append(w)
    return list(dict.fromkeys(found))
