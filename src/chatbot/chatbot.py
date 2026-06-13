"""
Chatbot Baseline — a plain LLM wrapper with NO tools and NO reasoning loop.

Purpose: demonstrate what a chatbot CAN'T do on multi-step queries so we can
appreciate what the ReAct Agent adds.
"""

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker

CHATBOT_SYSTEM_PROMPT = """You are a helpful e-commerce assistant.
Answer the user's questions as accurately as possible based solely on your
own knowledge. You do NOT have access to any external tools, live inventory,
or real-time data. If you don't know something, say so honestly.
"""


class Chatbot:
    """
    Minimal chatbot: one-shot LLM call, no tool use, no looping.
    Used as the performance baseline for comparison with ReActAgent.
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def run(self, user_input: str) -> str:
        """
        Send user_input to the LLM and return the response directly.
        No tools, no loops — just raw LLM capability.
        """
        logger.log_event("CHATBOT_START", {
            "input": user_input,
            "model": self.llm.model_name,
        })

        result = self.llm.generate(user_input, system_prompt=CHATBOT_SYSTEM_PROMPT)

        # Telemetry: track every request for later comparison
        tracker.track_request(
            provider=result.get("provider", "unknown"),
            model=self.llm.model_name,
            usage=result.get("usage", {}),
            latency_ms=result.get("latency_ms", 0),
        )

        logger.log_event("CHATBOT_END", {
            "latency_ms": result.get("latency_ms"),
            "total_tokens": result.get("usage", {}).get("total_tokens"),
        })

        return result["content"]
