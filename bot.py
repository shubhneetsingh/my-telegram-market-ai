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
    ADMIN_USER_IDS,
    ADMIN_USERNAMES,
)
from memory import (
    save_chat_message,
    get_recent_chat_history,
    update_user_profile,
    get_user_profile,
    clear_user_history,
    get_system_analytics,
    record_activity,
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
    user = update.effective_user
    if user:
        record_activity(str(user.id), display_name=user.first_name or "", username=user.username or "", user_message="/start")

    welcome_text = (
        "👋 **Welcome to Trade with Bebo!**\n\n"
        "I'm **Bebo**, your personal AI Market Assistant.\n\n"
        "Get real-time market intelligence combining:\n"
        "📊 Technical Analysis\n"
        "📰 Live Market News\n"
        "⚖️ Independent Risk Checks\n\n"
        "🔍 **Start here:**\n"
        "• `/analyze BTC` — Get a complete market analysis\n"
        "• `/price GOLD` — Check live price and key levels\n"
        "• `/risk 100 95 115` — Calculate position risk & R:R\n"
        "• `/news Fed rate decision` — Get the latest market catalysts\n\n"
        "💡 **Or simply ask me a question naturally:**\n"
        "\"*Should I buy BTC now?*\"\n"
        "\"*Why is Gold moving?*\"\n"
        "\"*Analyze NAS100 on 1H*\"\n\n"
        "⚠️ _Market analysis is for informational purposes and does not guarantee trading results._\n\n"
        "**Ready? Try:** `/analyze BTC`"
    )
    await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /help command."""
    user = update.effective_user
    if user:
        record_activity(str(user.id), display_name=user.first_name or "", username=user.username or "", user_message="/help")

    help_text = (
        "📖 **Market Assistant Command Guide**\n\n"
        "**1. Full Multi-Agent Trade Analysis:**\n"
        "• `/analyze NVDA` or ask: *\"Should I enter Gold long here?\"*\n"
        "  ➔ Pulls 1D/1H OHLCV candles, checks EMAs/RSI/ATR, passes through Technical Analyst and Adversarial Critic.\n\n"
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
    """Handles the /model or status command."""
    user = update.effective_user
    if user:
        record_activity(str(user.id), display_name=user.first_name or "", username=user.username or "", user_message="/model")

    status_text = (
        "🛡️ **System Architecture & Status**\n\n"
        "• **Status**: 🟢 Operational (24/7 Cloud)\n"
        "• **Architecture**: Multi-Agent Neural Pipeline\n"
        "• **Workflow**: `Router` ➔ `News Agent` ➔ `Technical Agent` ➔ `Adversarial Critic` ➔ `Synthesizer`\n"
        "• **Quantitative Math**: Live Multi-Timeframe OHLCV (1D / 1H) + Indicator Confluence\n"
        "• **Risk Engine**: Deterministic Position & Capital Protection Engine\n"
        "• **Memory**: Persistent User Profile & Adaptive Learning"
    )
    await update.message.reply_text(status_text, parse_mode=ParseMode.MARKDOWN)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /stats command — displays real-time user metrics (Admin Only)."""
    user_id = str(update.effective_user.id)
    username = (update.effective_user.username or "").lower().lstrip("@")

    # If admins are configured in .env, restrict access to admins only
    if ADMIN_USER_IDS or ADMIN_USERNAMES:
        is_admin = (user_id in ADMIN_USER_IDS) or (username in ADMIN_USERNAMES)
        if not is_admin:
            await update.message.reply_text(
                "🔒 **Access Restricted**: Analytics and user data are reserved for the bot administrator.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

    analytics = get_system_analytics()
    total_users = analytics["total_users"]
    total_messages = analytics["total_messages"]
    active_24h = analytics["active_24h"]
    recent_users = analytics["recent_users"]

    user_lines = []
    for i, u in enumerate(recent_users[:10], 1):
        name = u["display_name"] or "Trader"
        uname = f"(@{u['username']})" if u["username"] else ""
        msgs = u["msg_count"]
        user_lines.append(f"{i}. **{name}** {uname} — `{msgs} msgs`")

    users_str = "\n".join(user_lines) if user_lines else "_No user records yet._"

    report = (
        "📊 **Trade with Bebo — Live User Analytics**\n\n"
        f"👥 **Total Registered Users**: `{total_users}`\n"
        f"💬 **Total Messages Processed**: `{total_messages}`\n"
        f"⚡ **Active Traders (Last 24H)**: `{active_24h}`\n\n"
        "**Recent Active Users:**\n"
        f"{users_str}\n\n"
        "🌐 _Live Web Dashboard also accessible on your Render URL at `/stats`_"
    )
    await update.message.reply_text(report, parse_mode=ParseMode.MARKDOWN)


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
    user = update.effective_user
    if user:
        record_activity(str(user.id), display_name=user.first_name or "", username=user.username or "", user_message=f"/price {' '.join(context.args)}")

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
    user = update.effective_user
    if user:
        record_activity(str(user.id), display_name=user.first_name or "", username=user.username or "", user_message=f"/analyze {' '.join(context.args)}")

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
    user = update.effective_user
    if user:
        record_activity(str(user.id), display_name=user.first_name or "", username=user.username or "", user_message=f"/risk {' '.join(context.args)}")

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
    user = update.effective_user
    if user:
        record_activity(str(user.id), display_name=user.first_name or "", username=user.username or "", user_message=f"/news {' '.join(context.args)}")

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

    # Record user profile & message to SQLite
    record_activity(user_id, display_name=first_name, username=username, user_message=user_message)

    # Fetch persistent chat history from SQLite
    chat_history = get_recent_chat_history(user_id, limit=MAX_MEMORY_TURNS)

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
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("users", stats_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("✅ Multi-Agent Bot is online and polling Telegram! Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
