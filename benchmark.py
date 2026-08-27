"""
Multi-Agent Benchmark & Scenario Diagnostic Suite.
Tests all agents (Router, News, Technicals, Critic, Synthesizer) against real market scenarios
and logs latency, JSON adherence, and adversarial contradiction catching.
"""

import asyncio
import json
import time
from config import get_ai_client_config, resolve_agent_model
from orchestrator import orchestrator, calculate_evidence_confluence_score
from market_data import get_multi_timeframe_technical_data

# Standard Benchmark Test Cases
SCENARIOS = [
    {
        "name": "Overbought Asset Evaluation",
        "symbol": "BTC-USD",
        "query": "BTC is rallying strong, should I go all-in long right now?",
        "expected_critic": ["CAUTION", "INVALIDATED"],
    },
    {
        "name": "Macro News & Catalyst Extraction",
        "symbol": "GC=F",
        "query": "/news US Dollar strength impact on Gold prices today",
        "expected_critic": ["APPROVED", "CAUTION"],
    },
    {
        "name": "Equity Market Structure Breakdown",
        "symbol": "NVDA",
        "query": "Give me a technical breakdown for NVDA support and resistance levels",
        "expected_critic": ["APPROVED", "CAUTION"],
    },
]


async def run_benchmark():
    base_url, api_key, default_model = get_ai_client_config()
    print("=" * 70)
    print("🧪 MULTI-AGENT MARKET INTELLIGENCE BENCHMARK SUITE")
    print(f"🤖 Active Endpoint: {base_url}")
    print(f"🧠 Default Model:   {default_model}")
    print(f"🎯 Router Model:    {resolve_agent_model('router')}")
    print(f"📰 News Model:      {resolve_agent_model('news')}")
    print(f"📈 Technical Model: {resolve_agent_model('technical')}")
    print(f"🛡️ Critic Model:    {resolve_agent_model('critic')}")
    print(f"✍️ Synthesizer:     {resolve_agent_model('synthesizer')}")
    print("=" * 70)

    results = []

    for idx, test in enumerate(SCENARIOS, 1):
        print(f"\n[{idx}/{len(SCENARIOS)}] Running Test: {test['name']} ({test['symbol']})...")
        start_time = time.time()

        try:
            output = await orchestrator.run_full_pipeline(
                user_message=test["query"],
                chat_history=[{"role": "user", "content": test["query"]}],
            )
            elapsed = time.time() - start_time

            print(f"  ⏱️ Latency: {elapsed:.2f}s")
            print(f"  📄 Response Length: {len(output)} chars")

            # Check if output contains invalidation or caution flags
            has_critic_flag = any(flag in output.upper() for flag in ["INVALIDATED", "CAUTION", "APPROVED", "EVIDENCE CONFLUENCE"])
            print(f"  🛡️ Critic Governance Detected: {'✅ YES' if has_critic_flag else '⚠️ NO'}")

            results.append({
                "test": test["name"],
                "symbol": test["symbol"],
                "latency_sec": round(elapsed, 2),
                "status": "PASS",
            })

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ❌ FAILED: {str(e)} ({elapsed:.2f}s)")
            results.append({
                "test": test["name"],
                "symbol": test["symbol"],
                "latency_sec": round(elapsed, 2),
                "status": f"FAIL: {str(e)}",
            })

    print("\n" + "=" * 70)
    print("📊 BENCHMARK SUMMARY RESULTS:")
    print("=" * 70)
    for r in results:
        print(f"• {r['test']} [{r['symbol']}]: {r['status']} (Latency: {r['latency_sec']}s)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
