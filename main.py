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

from memory import get_system_analytics

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
    stats_command,
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
# FLASK HTTP HEALTH-CHECK & ANALYTICS DASHBOARD
# ==============================================================================
web_app = Flask(__name__)


@web_app.route("/")
def home():
    return (
        "<h1>📈 Trade with Bebo is ONLINE</h1>"
        "<p><strong>Assistant:</strong> Bebo (Multi-Agent Market AI)</p>"
        "<p><strong>Status:</strong> 🟢 24/7 Polling Operational</p>"
        "<p><a href='/stats'>👉 View Live User Analytics Dashboard</a></p>"
        f"<p><strong>Server UTC Time:</strong> {datetime.now(timezone.utc).isoformat()}</p>"
    )


@web_app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "service": "Trade with Bebo",
        "persona": "Bebo",
        "operational": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }), 200


@web_app.route("/stats")
def web_stats():
    analytics = get_system_analytics()
    rows = ""
    for i, u in enumerate(analytics["recent_users"], 1):
        uname = f"@{u['username']}" if u.get('username') else "N/A"
        name = u.get('display_name') or 'Trader'
        msgs = u.get('msg_count', 0)
        updated = u.get('updated_at', '')
        rows += f"<tr><td>{i}</td><td><strong>{name}</strong></td><td>{uname}</td><td><span class='badge'>{msgs} msgs</span></td><td>{updated}</td></tr>"

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Trade with Bebo — Live Analytics</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b0f19; color: #f8fafc; margin: 0; padding: 40px 20px; }}
            .container {{ max-width: 900px; margin: 0 auto; }}
            .header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 30px; }}
            h1 {{ margin: 0; font-size: 26px; color: #38bdf8; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 30px; }}
            .metric {{ background: #1e293b; border-radius: 14px; padding: 24px; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); }}
            .metric h3 {{ margin: 0; color: #94a3b8; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }}
            .metric p {{ margin: 12px 0 0 0; font-size: 34px; font-weight: 700; color: #f8fafc; }}
            .card {{ background: #1e293b; border-radius: 14px; padding: 24px; border: 1px solid #334155; }}
            .card h2 {{ margin: 0 0 20px 0; font-size: 18px; color: #f8fafc; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 14px 16px; text-align: left; border-bottom: 1px solid #334155; }}
            th {{ color: #94a3b8; font-size: 12px; text-transform: uppercase; font-weight: 600; }}
            .badge {{ background: #0284c7; color: #fff; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }}
            .status-dot {{ display: inline-block; width: 10px; height: 10px; background: #22c55e; border-radius: 50%; margin-right: 6px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <h1>📈 Trade with Bebo</h1>
                    <p style="margin: 6px 0 0 0; color: #94a3b8;"><span class="status-dot"></span>Live System & User Activity Dashboard</p>
                </div>
            </div>
            <div class="grid">
                <div class="metric"><h3>Total Registered Users</h3><p>{analytics['total_users']}</p></div>
                <div class="metric"><h3>Total Messages Processed</h3><p>{analytics['total_messages']}</p></div>
                <div class="metric"><h3>Active Traders (24H)</h3><p>{analytics['active_24h']}</p></div>
            </div>
            <div class="card">
                <h2>👥 Recent Active Traders</h2>
                <table>
                    <thead><tr><th>#</th><th>Name</th><th>Telegram Username</th><th>Activity</th><th>Last Active (UTC)</th></tr></thead>
                    <tbody>{rows or "<tr><td colspan='5' style='color:#94a3b8;'>No users recorded yet. Share the bot to see live metrics!</td></tr>"}</tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    return html


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
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("users", stats_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    logger.info("✅ Telegram Bot polling loop started. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
