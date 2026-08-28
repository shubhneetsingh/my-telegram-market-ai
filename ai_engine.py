"""
AI Intelligence Engine: Multi-provider client supporting NVIDIA NIM, Groq, OpenRouter, DeepSeek, and Ollama.
Supports per-agent model overrides and chain-of-thought cleaning.
"""

import logging
import re
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from config import get_ai_client_config, SYSTEM_PROMPT, AI_PROVIDER, resolve_agent_model

logger = logging.getLogger(__name__)


def create_ai_client() -> tuple[AsyncOpenAI, str]:
    """Instantiates the AsyncOpenAI client configured for the active provider."""
    base_url, api_key, model = get_ai_client_config()

    if not api_key and AI_PROVIDER != "ollama":
        logger.warning(
            f"⚠️ No API key found for provider '{AI_PROVIDER}'. "
            f"Please configure your key in the .env file."
        )

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key or "missing_key",
        timeout=20.0,
        max_retries=1,
    )
    return client, model


def clean_reasoning_output(text: str) -> str:
    """
    Cleans up internal chain-of-thought blocks like <think>...</think>
    and 'Here's a thinking process:' blocks from reasoning models.
    """
    if not text:
        return ""
    
    # 1. Clean <think>...</think> tags
    if "<think>" in text and "</think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 2. Clean 'Here's a thinking process:' preamble if present
    if "Here's a thinking process:" in text:
        # If there is a clear response after the thinking process
        parts = re.split(r"(?:### Final Output|Final Answer:|Final Response:|\n\n\*\*Final|\n\n---\n\n)", text, flags=re.IGNORECASE)
        if len(parts) > 1:
            text = parts[-1].strip()

    return text.strip()


class MarketAIEngine:
    def __init__(self):
        self.client, self.model = create_ai_client()

    def reload_client(self):
        """Refreshes client configuration."""
        self.client, self.model = create_ai_client()

    async def generate_response(
        self,
        chat_history: List[Dict[str, str]],
        live_market_context: Optional[str] = None,
        system_prompt: str = SYSTEM_PROMPT,
        model: Optional[str] = None,
    ) -> str:
        """Sends chat history to LLM with optional market context."""
        prompt_with_context = system_prompt

        if live_market_context:
            prompt_with_context += (
                f"\n\n========================================\n"
                f"REAL-TIME MARKET CONTEXT & LIVE DATA:\n"
                f"{live_market_context}\n"
                f"========================================\n"
                f"INSTRUCTION: Synthesize the above live data accurately into your response."
            )

        messages = [{"role": "system", "content": prompt_with_context}] + chat_history
        target_model = model or self.model
        fallback_models = [target_model, "openai/gpt-oss-20b", "meta/llama-3.2-11b-vision-instruct"]
        # Deduplicate while preserving order
        candidate_models = list(dict.fromkeys(fallback_models))

        last_error = ""
        for attempt_model in candidate_models:
            try:
                response = await self.client.chat.completions.create(
                    model=attempt_model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1000,
                )

                raw_content = response.choices[0].message.content or ""
                cleaned = clean_reasoning_output(raw_content)
                if cleaned:
                    return cleaned

            except Exception as e:
                last_error = str(e)
                logger.warning(f"AI Generation failed on {attempt_model}: {last_error}. Retrying fallback...")
                continue

        logger.error(f"All AI candidates failed. Last error: {last_error}")
        return f"⚠️ An error occurred while communicating with the AI engine ({AI_PROVIDER}):\n`{last_error}`"


# Global singleton instance
ai_engine = MarketAIEngine()
