"""
Deterministic Risk & Position Sizing Engine.
Performs exact mathematical calculations for risk management, position size, and R:R ratios.
"""

from typing import Dict, Any, Optional


def calculate_position_and_risk(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    account_balance: float = 10000.0,
    risk_percentage: float = 1.0,
    atr_14: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes exact trade parameters:
    - Dollar Risk
    - Share / Unit position size
    - Stop distance ($ and %)
    - Reward distance ($ and %)
    - Risk-to-Reward (R:R)
    - ATR stop adequacy check
    """
    if entry_price <= 0:
        raise ValueError("Entry price must be positive.")

    is_long = take_profit > entry_price

    if is_long:
        if stop_loss >= entry_price:
            raise ValueError("For a LONG trade, Stop Loss must be strictly below Entry Price.")
        stop_dist = entry_price - stop_loss
        reward_dist = take_profit - entry_price
    else:
        # Short trade
        if stop_loss <= entry_price:
            raise ValueError("For a SHORT trade, Stop Loss must be strictly above Entry Price.")
        stop_dist = stop_loss - entry_price
        reward_dist = entry_price - take_profit

    stop_pct = (stop_dist / entry_price) * 100
    reward_pct = (reward_dist / entry_price) * 100

    # Risk in dollar terms
    dollar_risk = account_balance * (risk_percentage / 100.0)

    # Position size in units / shares
    units = dollar_risk / stop_dist if stop_dist > 0 else 0.0
    position_value = units * entry_price

    # Risk-to-reward ratio
    risk_reward = reward_dist / stop_dist if stop_dist > 0 else 0.0

    # Dollar reward
    dollar_reward = units * reward_dist

    # ATR Stop Check
    atr_ratio = None
    atr_warning = None
    if atr_14 and atr_14 > 0:
        atr_ratio = stop_dist / atr_14
        if atr_ratio < 0.8:
            atr_warning = f"⚠️ Stop is very tight ({atr_ratio:.2f}x ATR). High risk of noise stop-out."
        elif atr_ratio > 3.0:
            atr_warning = f"⚠️ Stop is unusually wide ({atr_ratio:.2f}x ATR). May require smaller size."

    quality = "POOR (< 1:1.5)"
    if risk_reward >= 3.0:
        quality = "EXCELLENT (≥ 1:3.0)"
    elif risk_reward >= 2.0:
        quality = "HIGH QUALITY (≥ 1:2.0)"
    elif risk_reward >= 1.5:
        quality = "ACCEPTABLE (≥ 1:1.5)"

    return {
        "direction": "LONG" if is_long else "SHORT",
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "stop_distance": round(stop_dist, 4),
        "stop_pct": round(stop_pct, 2),
        "reward_distance": round(reward_dist, 4),
        "reward_pct": round(reward_pct, 2),
        "risk_reward_ratio": round(risk_reward, 2),
        "quality": quality,
        "account_balance": account_balance,
        "risk_percentage": risk_percentage,
        "dollar_risk": round(dollar_risk, 2),
        "dollar_reward": round(dollar_reward, 2),
        "recommended_units": round(units, 4),
        "total_position_value": round(position_value, 2),
        "atr_14": atr_14,
        "atr_ratio": round(atr_ratio, 2) if atr_ratio else None,
        "atr_warning": atr_warning,
    }


def format_risk_report(risk_data: Dict[str, Any]) -> str:
    """Formats calculated risk parameters into a clear Telegram markdown block."""
    dir_emoji = "🟢" if risk_data["direction"] == "LONG" else "🔴"
    lines = [
        f"🛡️ **DETERMINISTIC RISK & POSITION BREAKDOWN**",
        f"{dir_emoji} **Trade Direction**: `{risk_data['direction']}`",
        f"🎯 **Entry**: `${risk_data['entry_price']:,.2f}`",
        f"🛑 **Stop Loss**: `${risk_data['stop_loss']:,.2f}` (-{risk_data['stop_pct']}%)",
        f"🏆 **Take Profit**: `${risk_data['take_profit']:,.2f}` (+{risk_data['reward_pct']}%)",
        "",
        f"⚖️ **Risk : Reward Ratio**: `1 : {risk_data['risk_reward_ratio']}` ({risk_data['quality']})",
        f"💰 **Risk Amount ({risk_data['risk_percentage']}%)**: `${risk_data['dollar_risk']:,.2f}` (Target Reward: `${risk_data['dollar_reward']:,.2f}`)",
        f"📦 **Suggested Position Size**: `{risk_data['recommended_units']:,.4f}` units (${risk_data['total_position_value']:,.2f} total exposure)",
    ]

    if risk_data.get("atr_warning"):
        lines.append(f"\n{risk_data['atr_warning']}")

    return "\n".join(lines)
