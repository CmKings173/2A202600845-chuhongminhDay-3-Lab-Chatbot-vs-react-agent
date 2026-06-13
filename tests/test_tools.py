"""
Direct unit tests for all e-commerce tool functions.
No LLM involved — pure function testing.
"""

import pytest
from src.tools.ecommerce_tools import (
    check_stock,
    get_product_price,
    get_discount,
    calc_shipping,
    TOOL_REGISTRY,
)


class TestCheckStock:
    def test_known_item(self):
        result = check_stock("iphone")
        assert "15" in result

    def test_unknown_item(self):
        result = check_stock("flying_car")
        assert "not found" in result.lower()

    def test_case_insensitive(self):
        result = check_stock("IPHONE")
        assert "15" in result

    def test_item_with_spaces(self):
        result = check_stock("  macbook pro  ")
        assert "5" in result


class TestGetProductPrice:
    def test_known_item(self):
        result = get_product_price("iphone")
        assert "$999.00" in result

    def test_unknown_item(self):
        result = get_product_price("banana")
        assert "not found" in result.lower()


class TestGetDiscount:
    def test_valid_code(self):
        result = get_discount("WINNER")
        assert "10.0" in result

    def test_invalid_code(self):
        result = get_discount("NOPE")
        assert "invalid" in result.lower() or "expired" in result.lower()

    def test_uppercase_normalization(self):
        """Both uppercase and lowercase codes should return the same discount %."""
        result1 = get_discount("save20")
        result2 = get_discount("SAVE20")
        # The display may show original case, but both must contain 20.0% discount
        assert "20.0" in result1
        assert "20.0" in result2


class TestCalcShipping:
    def test_valid_destination(self):
        result = calc_shipping(2.0, "Hanoi")
        assert "$10.00" in result

    def test_unsupported_destination(self):
        result = calc_shipping(1.0, "Mars")
        assert "not supported" in result.lower()

    def test_zero_weight(self):
        result = calc_shipping(0, "Hanoi")
        assert "positive" in result.lower()

    def test_negative_weight(self):
        result = calc_shipping(-5, "Hanoi")
        assert "positive" in result.lower()


class TestToolRegistry:
    def test_all_tools_present(self):
        names = {t["name"] for t in TOOL_REGISTRY}
        assert "check_stock" in names
        assert "get_product_price" in names
        assert "get_discount" in names
        assert "calc_shipping" in names

    def test_all_tools_have_function(self):
        for tool in TOOL_REGISTRY:
            assert callable(tool["function"]), f"{tool['name']} missing callable 'function'"

    def test_all_tools_have_description(self):
        for tool in TOOL_REGISTRY:
            assert len(tool["description"]) > 20, f"{tool['name']} description too short"
