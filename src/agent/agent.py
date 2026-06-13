"""
ReAct Agent — Thought → Action → Observation loop.

ReAct = Reasoning + Acting.
Paper: "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)

How it works:
  1. We send the user query + system prompt to the LLM.
  2. The LLM responds with a "Thought" (reasoning) and an "Action" (tool call).
  3. We parse the Action, execute the real tool, get back an "Observation".
  4. We append the Observation to the conversation and call the LLM again.
  5. We repeat until the LLM writes "Final Answer:" or we hit max_steps.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


# ---------------------------------------------------------------------------
# Agent v2 system prompt  (improved over v1 — see change notes below)
# Key improvements vs v1:
#   - Added strict JSON format for Action to avoid freeform parsing failures
#   - Added Few-Shot examples so the LLM knows the exact argument format
#   - Added explicit "do not hallucinate tool names" warning
#   - Added instruction to stop and say Final Answer if a tool returns an error
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are a smart e-commerce assistant with access to real tools.
You must follow the ReAct format EXACTLY. Do not skip steps.

== Available Tools ==
{tool_descriptions}

== Format Rules (STRICT) ==
Always produce exactly one block per step:

Thought: <your reasoning about what to do next>
Action: {{"tool": "<tool_name>", "args": {{"<param1>": "<value1>", "<param2>": "<value2>"}}}}

After you receive an Observation, continue:

Thought: <reasoning using the observation>
Action: {{"tool": "<next_tool>", "args": {{...}}}}

When you have enough information to answer the user, write:

Final Answer: <your complete answer to the user>

== Rules ==
- ONLY use tool names from the list above. Never invent a tool.
- Action MUST be valid JSON. No markdown backticks around it.
- Each parameter value must match the tool's expected type (string / float).
- If a tool returns an error or unexpected result, adapt your strategy.
- Do NOT repeat the same Action twice with identical arguments.

== Few-Shot Example ==
User: How many MacBooks do we have and what is the price?

Thought: I need to check the stock for MacBook first.
Action: {{"tool": "check_stock", "args": {{"item_name": "macbook pro"}}}}

Observation: 'macbook pro' has 5 units available in stock.

Thought: Now I need to get the price.
Action: {{"tool": "get_product_price", "args": {{"item_name": "macbook pro"}}}}

Observation: The unit price of 'macbook pro' is $1999.00 USD.

Thought: I have all the information needed.
Final Answer: We have 5 MacBook Pros in stock at $1999.00 each.
"""


class ReActAgent:
    """
    ReAct-style Agent implementing the Thought-Action-Observation loop.

    Args:
        llm:       Any LLMProvider instance (OpenAI / Gemini / Local).
        tools:     List of tool dicts — each must have 'name', 'description',
                   and 'function' (callable).
        max_steps: Hard limit on Thought/Action cycles to prevent infinite loops.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: List[Dict[str, Any]],
        max_steps: int = 6,
    ):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        # Index tools by name for O(1) lookup
        self._tool_map: Dict[str, Dict] = {t["name"]: t for t in tools}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        """
        Execute the ReAct loop for a user query.

        Returns the Final Answer string, or an error message if max_steps
        is reached without a Final Answer.
        """
        logger.log_event("AGENT_START", {
            "input": user_input,
            "model": self.llm.model_name,
            "max_steps": self.max_steps,
        })

        # The "scratchpad" accumulates the entire Thought/Action/Observation
        # history so the LLM always sees the full context on each call.
        scratchpad = f"User: {user_input}\n\n"
        steps = 0
        final_answer: Optional[str] = None
        error_code: Optional[str] = None

        while steps < self.max_steps:
            # ── Step 1: call LLM ──────────────────────────────────────
            result = self.llm.generate(
                prompt=scratchpad,
                system_prompt=self._build_system_prompt(),
            )

            # Track telemetry for every LLM call
            tracker.track_request(
                provider=result.get("provider", "unknown"),
                model=self.llm.model_name,
                usage=result.get("usage", {}),
                latency_ms=result.get("latency_ms", 0),
            )

            llm_output = result["content"].strip()
            logger.log_event("AGENT_LLM_OUTPUT", {
                "step": steps + 1,
                "output_preview": llm_output[:300],
            })

            # ── Step 2: check for Final Answer ────────────────────────
            final = self._parse_final_answer(llm_output)
            if final:
                final_answer = final
                scratchpad += llm_output + "\n"
                break

            # ── Step 3: parse the Action ──────────────────────────────
            tool_name, tool_args, parse_error = self._parse_action(llm_output)

            if parse_error or tool_name is None:
                # JSON parse failure → log, append feedback, continue
                error_code = "JSON_PARSE_ERROR"
                logger.log_event("AGENT_ERROR", {
                    "step": steps + 1,
                    "error_code": error_code,
                    "raw_output": llm_output[:500],
                })
                observation = (
                    "ERROR: Could not parse your Action as valid JSON. "
                    "Please rewrite it as: "
                    '{"tool": "<name>", "args": {"<param>": "<value>"}}'
                )
                scratchpad += llm_output + f"\nObservation: {observation}\n\n"
                steps += 1
                continue

            # ── Step 4: execute tool ──────────────────────────────────
            observation = self._execute_tool(tool_name, tool_args)
            logger.log_event("AGENT_TOOL_CALL", {
                "step": steps + 1,
                "tool": tool_name,
                "args": tool_args,
                "observation": observation,
            })

            # Append Observation back to scratchpad so LLM sees it next round
            scratchpad += llm_output + f"\nObservation: {observation}\n\n"
            steps += 1

        # ── Post-loop handling ─────────────────────────────────────────
        if final_answer is None:
            error_code = "MAX_STEPS_EXCEEDED"
            final_answer = (
                f"I reached the step limit ({self.max_steps}) without "
                "arriving at a Final Answer. The conversation so far:\n"
                + scratchpad
            )

        logger.log_event("AGENT_END", {
            "steps": steps,
            "success": error_code is None,
            "error_code": error_code,
        })

        return final_answer

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build the system prompt by injecting the current tool list."""
        tool_descriptions = "\n".join(
            f"  - {t['name']}: {t['description']}"
            for t in self.tools
        )
        return SYSTEM_PROMPT_TEMPLATE.format(tool_descriptions=tool_descriptions)

    # ── Parsers ────────────────────────────────────────────────────────

    def _parse_final_answer(self, text: str) -> Optional[str]:
        """
        Extract the text after 'Final Answer:' if present.
        Handles variations like 'Final Answer :' or on a new line.
        """
        match = re.search(
            r"Final\s+Answer\s*:\s*(.+)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return None

    def _parse_action(
        self, text: str
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
        """
        Parse the Action block from LLM output.

        Expected format (v2 — strict JSON):
            Action: {"tool": "check_stock", "args": {"item_name": "iphone"}}

        Returns: (tool_name, args_dict, error_message)
        On success:  (str, dict, None)
        On failure:  (None, None, str)
        """
        # 1. Find the "Action:" line
        action_match = re.search(
            r"Action\s*:\s*(.+?)(?:\nObservation|\nThought|\nFinal|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if not action_match:
            return None, None, "No Action block found in LLM output."

        raw = action_match.group(1).strip()

        # 2. Strip markdown code fences if the LLM wrapped JSON in ```
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        # 3. Parse JSON
        try:
            action_dict = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, None, f"JSONDecodeError: {exc} — raw: {raw[:200]}"

        # 4. Validate required keys
        tool_name = action_dict.get("tool")
        args = action_dict.get("args", {})

        if not tool_name:
            return None, None, "Action JSON missing 'tool' key."
        if not isinstance(args, dict):
            return None, None, "Action JSON 'args' must be a dict."

        return tool_name, args, None

    # ── Tool execution ─────────────────────────────────────────────────

    def _execute_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> str:
        """
        Look up tool by name and call it with the parsed args.

        Handles:
        - Tool not found (hallucinated name)
        - Wrong argument types (e.g., LLM passes string where float expected)
        - Runtime exceptions inside the tool
        """
        tool = self._tool_map.get(tool_name)

        if tool is None:
            available = ", ".join(self._tool_map.keys())
            logger.log_event("AGENT_ERROR", {
                "error_code": "HALLUCINATED_TOOL",
                "tool_name": tool_name,
                "available_tools": available,
            })
            return (
                f"ERROR: Tool '{tool_name}' does not exist. "
                f"Available tools: {available}. Please use a valid tool name."
            )

        try:
            # Dynamically call the tool's function with the parsed args dict
            fn = tool["function"]
            result = fn(**args)
            return str(result)
        except TypeError as exc:
            # Wrong parameter names or count
            logger.log_event("AGENT_ERROR", {
                "error_code": "TOOL_ARGUMENT_ERROR",
                "tool": tool_name,
                "args": args,
                "error": str(exc),
            })
            return (
                f"ERROR calling '{tool_name}': {exc}. "
                f"Check the argument names and types."
            )
        except Exception as exc:
            logger.log_event("AGENT_ERROR", {
                "error_code": "TOOL_RUNTIME_ERROR",
                "tool": tool_name,
                "error": str(exc),
            })
            return f"ERROR: '{tool_name}' raised an exception: {exc}"
