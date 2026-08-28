"""
24/7 Cloud Entrypoint for Render / Railway / VPS with UptimeRobot Keep-Alive.
Runs a lightweight Flask HTTP health-check server concurrently with the Telegram Bot.
"""

import os
import sys
import threading
import logging
from datetime import datetime, timezone
from flask import Flask, jsonify

from bot import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    TELEGRAM_TOKEN,
    AI_PROVIDER,
    orchestrator,
    start_command,
    help_command,
    analyze_command,
    price_command,
    risk_command,
    news_command,
    model_command,
    clear_command,
    handle_message,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("main_runner")

# ==============================================================================
# FLASK HTTP HEALTH-CHECK SERVER (For UptimeRobot Keep-Alive)
# ==============================================================================
web_app = Flask(__name__)


@web_app.route("/")
def home():
    return (
        "<h1>🤖 Multi-Agent Telegram AI Market Assistant is ONLINE</h1>"
        "<p><strong>Architecture:</strong> Multi-Agent Neural Pipeline</p>"
        "<p><strong>Status:</strong> 🟢 24/7 Polling Operational</p>"
        f"<p><strong>Server UTC Time:</strong> {datetime.now(timezone.utc).isoformat()}</p>"
    )


@web_app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "service": "Multi-Agent Telegram Market Assistant",
        "operational": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }), 200


def run_http_server():
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🌐 Starting HTTP Health Server on port {port} (for UptimeRobot keep-alive)...")
    web_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


# ==============================================================================
# MAIN 24/7 ENTRYPOINT
# ==============================================================================
def main():
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "your_telegram_bot_token_here":
        print("=" * 70)
        print("❌ ERROR: TELEGRAM_TOKEN is not configured!")
        print("Please check your environment variables or .env file.")
        print("=" * 70)
        sys.exit(1)

    # 1. Start Flask HTTP Server in a background daemon thread
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()

    # 2. Build Telegram Application
    print("=" * 70)
    print("🚀 Starting 24/7 Multi-Agent Telegram AI Market Assistant...")
    print(f"🤖 AI Provider:  {AI_PROVIDER.upper()}")
    print(f"🧠 Model:        {orchestrator.ai.model}")
    print("=" * 70)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("price", price_command))
    app.add_handler(CommandHandler("risk", risk_command))
    app.add_handler(CommandHandler("news", news_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    logger.info("✅ Telegram Bot polling loop started. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
