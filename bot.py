"""
Telegram AI Market Assistant Bot (Multi-Agent Version)
Integrates with Multi-Agent Orchestrator, Deterministic Technicals, and Risk Engine.
"""

import logging
import sys
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from config import (
    TELEGRAM_TOKEN,
    AI_PROVIDER,
    AI_MODEL,
    MAX_MEMORY_TURNS,
)
from orchestrator import orchestrator
from market_data import (
    get_multi_timeframe_technical_data,
    format_ticker_summary,
    search_live_market_news,
)
from risk_engine import calculate_position_and_risk, format_risk_report

# Configure logging
logging.basicConfig(
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# COMMAND HANDLERS
# ---------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /start command."""
    welcome_text = (
        "👋 **Welcome to your Multi-Agent AI Market Assistant!**\n\n"
        f"🤖 **Active Engine**: `{AI_PROVIDER.upper()}` (`{orchestrator.ai.model}`)\n"
        "⚡ **Pipeline**: `Intent Router` ➔ `Quant OHLCV Engine` ➔ `News Agent` ➔ `Technical Agent` ➔ `Adversarial Critic` ➔ `Synthesizer`\n\n"
        "**Key Commands:**\n"
        "• `/analyze <symbol>` — Full multi-agent trade thesis & risk review (e.g. `/analyze NVDA`, `/analyze BTC`)\n"
        "• `/price <symbol>` — Instant deterministic price & indicator metrics\n"
        "• `/risk <entry> <stop> <tp>` — Exact position sizing & R:R calculator (e.g. `/risk 100 95 115`)\n"
        "• `/news <query>` — Macro news catalyst & sentiment synthesis\n"
        "• `/model` — View active AI configuration\n"
        "• `/clear` — Reset memory\n"
        "• `/help` — Full guide\n\n"
        "💡 *Or simply ask any market question, trade setup, or chart pattern in natural language!*"
    )
    await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /help command."""
    help_text = (
        "📖 **Multi-Agent Market Assistant Guide**\n\n"
        "**1. Full Multi-Agent Trade Analysis:**\n"
        "• `/analyze NVDA` or ask: *\"Should I enter Gold long here?\"*\n"
        "  ➔ Pulls 1D/1H OHLCV candles, checks EMAs/RSI/ATR, passes through the Technical Analyst and the Adversarial Critic.\n\n"
        "**2. Fast Ticker Stats & Indicators:**\n"
        "• `/price BTC` or `/price EURUSD` or `/price SPX`\n\n"
        "**3. Deterministic Risk & Position Calculator:**\n"
        "• `/risk 100 95 115` (Entry: $100, Stop: $95, Target: $115)\n"
        "• `/risk 78000 76500 82000 25000 1.5` (With $25,000 balance & 1.5% risk)\n\n"
        "**4. Macro & News Engine:**\n"
        "• `/news US CPI report` or `/news Fed interest rates`\n\n"
        "**5. Reset Memory:**\n"
        "• `/clear` to start fresh."
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /model command."""
    status_text = (
        "🤖 **Multi-Agent AI Configuration**\n\n"
        f"• **Provider**: `{AI_PROVIDER}`\n"
        f"• **Base Model**: `{orchestrator.ai.model}`\n"
        f"• **Orchestration**: `Router` ➔ `News Agent` ➔ `Technical Agent` ➔ `Critic Agent` ➔ `Synthesizer`\n"
        f"• **Deterministic Math**: `yfinance (1D/1H OHLCV)` + `Python Risk Engine`\n"
        f"• **Memory Window**: `{MAX_MEMORY_TURNS}` turns\n\n"
        "To switch models or providers, update `AI_PROVIDER` and `AI_MODEL` in `.env`."
    )
    await update.message.reply_text(status_text, parse_mode=ParseMode.MARKDOWN)


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Resets conversational memory in SQLite."""
    user_id = str(update.effective_user.id)
    clear_user_history(user_id)
    context.user_data["chat_history"] = []
    await update.message.reply_text(
        "🧹 **Conversation memory cleared.** Let's start fresh!",
        parse_mode=ParseMode.MARKDOWN,
    )


async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles /price <ticker>."""
    if not context.args:
        await update.message.reply_text(
            "⚠️ Please specify a ticker symbol. Example: `/price NVDA` or `/price BTC`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    symbol = context.args[0].upper()
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    quote_data = await get_multi_timeframe_technical_data(symbol)
    if not quote_data:
        await update.message.reply_text(
            f"❌ Could not retrieve market data for `{symbol}`. Please check the ticker.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    summary = format_ticker_summary(quote_data)
    await update.message.reply_text(summary, parse_mode=ParseMode.MARKDOWN)


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles /analyze <ticker> — triggers full multi-agent pipeline."""
    if not context.args:
        await update.message.reply_text(
            "⚠️ Please specify an asset to analyze. Example: `/analyze NVDA` or `/analyze BTC`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    symbol = context.args[0].upper()
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    query = f"Provide a complete multi-agent trade analysis and risk breakdown for {symbol}."
    reply = await orchestrator.run_full_pipeline(query, [{"role": "user", "content": query}])

    try:
        await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text(reply)


async def risk_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles /risk <entry> <stop> <target> [balance] [risk_pct]
    Example: /risk 100 95 115 10000 1.0
    """
    if len(context.args) < 3:
        await update.message.reply_text(
            "⚠️ **Usage**: `/risk <entry> <stop_loss> <take_profit> [balance] [risk_pct]`\n"
            "Example: `/risk 100 95 115`\n"
            "Example with custom balance: `/risk 100 95 115 25000 1.5`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    try:
        entry = float(context.args[0])
        stop = float(context.args[1])
        tp = float(context.args[2])
        balance = float(context.args[3]) if len(context.args) > 3 else 10000.0
        risk_pct = float(context.args[4]) if len(context.args) > 4 else 1.0

        risk_data = calculate_position_and_risk(
            entry_price=entry,
            stop_loss=stop,
            take_profit=tp,
            account_balance=balance,
            risk_percentage=risk_pct,
        )
        report = format_risk_report(risk_data)
        await update.message.reply_text(report, parse_mode=ParseMode.MARKDOWN)
    except ValueError as e:
        await update.message.reply_text(f"⚠️ Calculation Error: {str(e)}", parse_mode=ParseMode.MARKDOWN)


async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles /news <query>."""
    if not context.args:
        await update.message.reply_text(
            "⚠️ Please specify a query. Example: `/news Fed interest rates`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    query = " ".join(context.args)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    reply = await orchestrator.run_full_pipeline(f"/news {query}", [{"role": "user", "content": query}])
    try:
        await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text(reply)


from memory import (
    save_chat_message,
    get_recent_chat_history,
    update_user_profile,
    get_user_profile,
    clear_user_history,
)


# ---------------------------------------------------------
# CONVERSATIONAL CHAT HANDLER (MULTI-AGENT PIPELINE)
# ---------------------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Routes conversational questions through the Multi-Agent Orchestrator with Persistent Memory."""
    user_message = update.message.text
    if not user_message:
        return

    user_id = str(update.effective_user.id)
    username = update.effective_user.username or ""
    first_name = update.effective_user.first_name or ""

    # Update profile display name & username
    update_user_profile(user_id, display_name=first_name, username=username)

    # Fetch persistent chat history from SQLite
    chat_history = get_recent_chat_history(user_id, limit=MAX_MEMORY_TURNS)

    # Save user message to persistent DB
    save_chat_message(user_id, "user", user_message)

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    # Run through Multi-Agent Orchestrator
    reply = await orchestrator.run_full_pipeline(user_message, chat_history)

    # Save assistant response to persistent DB
    save_chat_message(user_id, "assistant", reply)

    # Send response
    try:
        await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text(reply)


# ---------------------------------------------------------
# MAIN RUNNER
# ---------------------------------------------------------
def main():
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "your_telegram_bot_token_here":
        print("=" * 70)
        print("❌ ERROR: TELEGRAM_TOKEN is not configured!")
        print("Please edit the '.env' file in this directory and set:")
        print("TELEGRAM_TOKEN=your_token_from_botfather")
        print("=" * 70)
        sys.exit(1)

    print("=" * 70)
    print("🚀 Starting Multi-Agent Telegram AI Market Assistant...")
    print(f"🤖 AI Provider:     {AI_PROVIDER.upper()}")
    print(f"🧠 Model:           {orchestrator.ai.model}")
    print("⚡ Pipeline:        Router -> Quant OHLCV -> News -> Technical -> Critic -> Synthesizer")
    print("=" * 70)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("price", price_command))
    app.add_handler(CommandHandler("risk", risk_command))
    app.add_handler(CommandHandler("news", news_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("✅ Multi-Agent Bot is online and polling Telegram! Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
