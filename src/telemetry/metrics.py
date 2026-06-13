"""
Performance Tracker — industry-grade telemetry for LLM calls.

Tracks per-request and aggregate metrics:
  - Token counts (prompt / completion / total)
  - Latency (ms)
  - Cost estimate (USD) — per real pricing tables
  - Token efficiency ratio (completion / prompt)
  - Aggregate stats across a session
"""

from typing import Dict, Any, List, Optional
from src.telemetry.logger import logger


# ---------------------------------------------------------------------------
# Real pricing tables (USD per 1 000 tokens) — as of mid-2024
# ---------------------------------------------------------------------------
_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o":              {"prompt": 0.005,   "completion": 0.015},
    "gpt-4o-mini":         {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4-turbo":         {"prompt": 0.01,    "completion": 0.03},
    "gpt-3.5-turbo":       {"prompt": 0.0005,  "completion": 0.0015},
    # Google
    "gemini-1.5-flash":    {"prompt": 0.000075, "completion": 0.0003},
    "gemini-1.5-pro":      {"prompt": 0.00125,  "completion": 0.005},
    # Local — no cost
    "local":               {"prompt": 0.0,      "completion": 0.0},
}

_DEFAULT_PRICING = {"prompt": 0.01, "completion": 0.03}   # conservative fallback


def _calculate_cost(model: str, usage: Dict[str, int]) -> float:
    """
    Calculate real estimated cost in USD based on model pricing.

    Falls back to conservative defaults for unknown models.
    """
    # Normalize: strip provider prefix if present (e.g. "openai/gpt-4o")
    model_key = model.lower().split("/")[-1]

    pricing = _PRICING.get(model_key, _DEFAULT_PRICING)

    prompt_cost = (usage.get("prompt_tokens", 0) / 1_000) * pricing["prompt"]
    completion_cost = (usage.get("completion_tokens", 0) / 1_000) * pricing["completion"]
    return round(prompt_cost + completion_cost, 8)


class PerformanceTracker:
    """
    Tracks LLM call metrics across a session.

    Usage:
        tracker.track_request(provider, model, usage, latency_ms)
        summary = tracker.get_summary()
    """

    def __init__(self):
        self.session_metrics: List[Dict[str, Any]] = []

    def track_request(
        self,
        provider: str,
        model: str,
        usage: Dict[str, int],
        latency_ms: int,
        context: Optional[str] = None,     # "chatbot" | "agent" | etc.
    ) -> None:
        """
        Record a single LLM request with full metrics.

        Args:
            provider:    "openai" | "google" | "local"
            model:       Model name string
            usage:       Dict with prompt_tokens, completion_tokens, total_tokens
            latency_ms:  Wall-clock time for this call in milliseconds
            context:     Optional label (e.g. "agent_step_2")
        """
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens)

        # Token efficiency: how much useful output per token of input
        efficiency_ratio = (
            round(completion_tokens / prompt_tokens, 4)
            if prompt_tokens > 0 else 0.0
        )

        cost = _calculate_cost(model, usage)

        metric: Dict[str, Any] = {
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost,
            "token_efficiency_ratio": efficiency_ratio,  # completion / prompt
            **({"context": context} if context else {}),
        }

        self.session_metrics.append(metric)
        logger.log_event("LLM_METRIC", metric)

    def get_summary(self) -> Dict[str, Any]:
        """
        Aggregate metrics across all tracked requests in this session.

        Returns a dict suitable for printing or logging as a dashboard.
        """
        if not self.session_metrics:
            return {"message": "No requests tracked yet."}

        n = len(self.session_metrics)
        latencies = [m["latency_ms"] for m in self.session_metrics]
        total_tokens_list = [m["total_tokens"] for m in self.session_metrics]
        costs = [m["cost_usd"] for m in self.session_metrics]

        latencies_sorted = sorted(latencies)

        def percentile(data: list, p: int) -> float:
            idx = max(0, int(len(data) * p / 100) - 1)
            return data[idx]

        summary = {
            "total_requests": n,
            "total_prompt_tokens": sum(m["prompt_tokens"] for m in self.session_metrics),
            "total_completion_tokens": sum(m["completion_tokens"] for m in self.session_metrics),
            "total_tokens": sum(total_tokens_list),
            "avg_tokens_per_request": round(sum(total_tokens_list) / n, 1),
            "total_cost_usd": round(sum(costs), 6),
            "avg_cost_per_request_usd": round(sum(costs) / n, 6),
            "latency_avg_ms": round(sum(latencies) / n, 1),
            "latency_p50_ms": percentile(latencies_sorted, 50),
            "latency_p95_ms": percentile(latencies_sorted, 95),
            "latency_p99_ms": percentile(latencies_sorted, 99),
            "latency_max_ms": max(latencies),
        }

        logger.log_event("SESSION_SUMMARY", summary)
        return summary

    def reset(self) -> None:
        """Clear all tracked metrics (start a new session)."""
        self.session_metrics = []


# Global singleton — used by both Chatbot and ReActAgent
tracker = PerformanceTracker()
