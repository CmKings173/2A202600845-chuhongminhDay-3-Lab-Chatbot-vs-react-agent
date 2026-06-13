"""
Unit tests for ReActAgent — tool parsing, tool execution, loop logic.

Run with:
    pytest tests/test_agent.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from src.agent.agent import ReActAgent
from src.tools.ecommerce_tools import TOOL_REGISTRY


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_mock_llm(responses: list) -> MagicMock:
    """
    Create a mock LLMProvider that returns responses in sequence.
    Each call to generate() consumes the next item in `responses`.
    """
    llm = MagicMock()
    llm.model_name = "mock-model"
    llm.generate.side_effect = [
        {
            "content": text,
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            "latency_ms": 200,
            "provider": "mock",
        }
        for text in responses
    ]
    return llm


def make_agent(llm) -> ReActAgent:
    return ReActAgent(llm=llm, tools=TOOL_REGISTRY, max_steps=6)


# ---------------------------------------------------------------------------
# 1. Parser tests
# ---------------------------------------------------------------------------

class TestParseAction:
    def setup_method(self):
        llm = MagicMock()
        llm.model_name = "test"
        self.agent = ReActAgent(llm=llm, tools=TOOL_REGISTRY, max_steps=5)

    def test_valid_json_action(self):
        text = 'Thought: I need to check stock.\nAction: {"tool": "check_stock", "args": {"item_name": "iphone"}}'
        name, args, err = self.agent._parse_action(text)
        assert err is None
        assert name == "check_stock"
        assert args == {"item_name": "iphone"}

    def test_action_with_markdown_backticks(self):
        """LLM sometimes wraps JSON in ```json ... ``` — parser must strip it."""
        text = 'Action: ```json\n{"tool": "get_discount", "args": {"coupon_code": "WINNER"}}\n```'
        name, args, err = self.agent._parse_action(text)
        assert err is None
        assert name == "get_discount"
        assert args["coupon_code"] == "WINNER"

    def test_missing_action_block(self):
        text = "Thought: I am thinking but no action here."
        name, args, err = self.agent._parse_action(text)
        assert name is None
        assert err is not None

    def test_malformed_json(self):
        text = "Action: {tool: check_stock, args: {}}"   # not valid JSON
        name, args, err = self.agent._parse_action(text)
        assert name is None
        assert "JSONDecodeError" in err or err is not None

    def test_missing_tool_key(self):
        text = 'Action: {"action": "check_stock", "params": {}}'
        name, args, err = self.agent._parse_action(text)
        assert name is None

    def test_multi_arg_action(self):
        text = 'Action: {"tool": "calc_shipping", "args": {"weight_kg": 1.5, "destination": "Hanoi"}}'
        name, args, err = self.agent._parse_action(text)
        assert err is None
        assert name == "calc_shipping"
        assert args["destination"] == "Hanoi"
        assert args["weight_kg"] == 1.5


class TestParseFinalAnswer:
    def setup_method(self):
        llm = MagicMock()
        llm.model_name = "test"
        self.agent = ReActAgent(llm=llm, tools=TOOL_REGISTRY, max_steps=5)

    def test_final_answer_found(self):
        text = "Thought: I have all info.\nFinal Answer: The total is $50."
        result = self.agent._parse_final_answer(text)
        assert result == "The total is $50."

    def test_final_answer_case_insensitive(self):
        text = "final answer: Done!"
        result = self.agent._parse_final_answer(text)
        assert result == "Done!"

    def test_no_final_answer(self):
        text = "Thought: Still thinking.\nAction: {}"
        result = self.agent._parse_final_answer(text)
        assert result is None


# ---------------------------------------------------------------------------
# 2. Tool execution tests
# ---------------------------------------------------------------------------

class TestExecuteTool:
    def setup_method(self):
        llm = MagicMock()
        llm.model_name = "test"
        self.agent = ReActAgent(llm=llm, tools=TOOL_REGISTRY, max_steps=5)

    def test_check_stock_known_item(self):
        result = self.agent._execute_tool("check_stock", {"item_name": "iphone"})
        assert "15" in result
        assert "iphone" in result.lower()

    def test_get_product_price(self):
        result = self.agent._execute_tool("get_product_price", {"item_name": "macbook pro"})
        assert "1999" in result

    def test_get_discount_valid_code(self):
        result = self.agent._execute_tool("get_discount", {"coupon_code": "WINNER"})
        assert "10.0" in result

    def test_get_discount_invalid_code(self):
        result = self.agent._execute_tool("get_discount", {"coupon_code": "FAKE"})
        assert "invalid" in result.lower() or "expired" in result.lower()

    def test_calc_shipping(self):
        result = self.agent._execute_tool(
            "calc_shipping", {"weight_kg": 2.0, "destination": "Hanoi"}
        )
        assert "10.00" in result   # 2kg * $5/kg

    def test_hallucinated_tool(self):
        result = self.agent._execute_tool("nonexistent_tool", {})
        assert "ERROR" in result
        assert "nonexistent_tool" in result

    def test_wrong_arguments(self):
        """Passing wrong kwarg name should produce a clear error, not crash."""
        result = self.agent._execute_tool("check_stock", {"wrong_param": "iphone"})
        assert "ERROR" in result


# ---------------------------------------------------------------------------
# 3. Full ReAct loop tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestReActLoop:
    def test_single_tool_call_then_final_answer(self):
        """Agent calls one tool and returns Final Answer in 2 steps."""
        llm = make_mock_llm([
            # Step 1: LLM produces a Thought + Action
            'Thought: I need to check stock.\nAction: {"tool": "check_stock", "args": {"item_name": "iphone"}}',
            # Step 2: LLM sees Observation and gives Final Answer
            "Thought: I have the info.\nFinal Answer: iPhone has 15 units in stock.",
        ])
        agent = make_agent(llm)
        answer = agent.run("How many iPhones do we have?")

        assert "15" in answer
        assert llm.generate.call_count == 2

    def test_final_answer_on_first_response(self):
        """If LLM answers immediately (no tools needed), return directly."""
        llm = make_mock_llm([
            "Thought: Simple question.\nFinal Answer: We are open 9am-5pm.",
        ])
        agent = make_agent(llm)
        answer = agent.run("What are your business hours?")
        assert "9am" in answer
        assert llm.generate.call_count == 1

    def test_max_steps_exceeded(self):
        """Agent should stop gracefully if LLM never produces Final Answer."""
        llm = make_mock_llm(
            ['Thought: Thinking...\nAction: {"tool": "check_stock", "args": {"item_name": "x"}}'] * 10
        )
        agent = ReActAgent(llm=llm, tools=TOOL_REGISTRY, max_steps=3)
        answer = agent.run("Tell me something.")
        assert "step limit" in answer.lower() or "3" in answer

    def test_json_parse_error_recovery(self):
        """Agent should recover from one bad JSON response and continue."""
        llm = make_mock_llm([
            # Bad output — invalid JSON
            "Thought: Let me check.\nAction: check_stock(iphone)",
            # LLM fixes itself after error feedback
            'Thought: Fix the format.\nAction: {"tool": "check_stock", "args": {"item_name": "iphone"}}',
            # Final answer
            "Thought: Got it.\nFinal Answer: 15 units available.",
        ])
        agent = make_agent(llm)
        answer = agent.run("How many iPhones?")
        assert "15" in answer

    def test_hallucinated_tool_recovery(self):
        """Agent should handle hallucinated tool name and keep going."""
        llm = make_mock_llm([
            # Hallucinates a tool
            'Thought: Let me search.\nAction: {"tool": "web_search", "args": {"query": "iphone price"}}',
            # Gets error observation, corrects itself
            'Thought: Use real tool.\nAction: {"tool": "get_product_price", "args": {"item_name": "iphone"}}',
            # Final answer
            "Final Answer: iPhone price is $999.",
        ])
        agent = make_agent(llm)
        answer = agent.run("What is the iPhone price?")
        assert "$999" in answer or "999" in answer

    def test_multi_step_ecommerce_query(self):
        """
        Complex test: buy 2 iPhones with coupon WINNER, ship to Hanoi.
        Agent must call 4 tools: check_stock, get_product_price, get_discount, calc_shipping.
        """
        llm = make_mock_llm([
            'Thought: Check stock first.\nAction: {"tool": "check_stock", "args": {"item_name": "iphone"}}',
            'Thought: Get price.\nAction: {"tool": "get_product_price", "args": {"item_name": "iphone"}}',
            'Thought: Get discount.\nAction: {"tool": "get_discount", "args": {"coupon_code": "WINNER"}}',
            'Thought: Calc shipping.\nAction: {"tool": "calc_shipping", "args": {"weight_kg": 0.5, "destination": "Hanoi"}}',
            "Final Answer: 2 iPhones with WINNER discount + shipping to Hanoi = $1800.05 total.",
        ])
        agent = make_agent(llm)
        answer = agent.run("I want to buy 2 iPhones with coupon WINNER and ship to Hanoi.")
        assert llm.generate.call_count == 5
        assert answer is not None


# ---------------------------------------------------------------------------
# 4. Tool function unit tests
# ---------------------------------------------------------------------------

class TestEcommerceTools:
    def test_check_stock_out_of_stock(self):
        from src.tools.ecommerce_tools import check_stock
        # Override stock to 0 just for this test
        import src.tools.ecommerce_tools as module
        original = module._STOCK_DB.copy()
        module._STOCK_DB["test_item"] = 0
        result = check_stock("test_item")
        assert "out of stock" in result.lower()
        module._STOCK_DB.clear()
        module._STOCK_DB.update(original)

    def test_calc_shipping_unsupported_city(self):
        from src.tools.ecommerce_tools import calc_shipping
        result = calc_shipping(1.0, "Mars")
        assert "not supported" in result.lower()

    def test_calc_shipping_invalid_weight(self):
        from src.tools.ecommerce_tools import calc_shipping
        result = calc_shipping(-1, "Hanoi")
        assert "positive" in result.lower()

    def test_get_discount_case_insensitive(self):
        from src.tools.ecommerce_tools import get_discount
        result_upper = get_discount("WINNER")
        result_lower = get_discount("winner")
        # Both should return the same discount
        assert "10.0" in result_upper
        assert "10.0" in result_lower
