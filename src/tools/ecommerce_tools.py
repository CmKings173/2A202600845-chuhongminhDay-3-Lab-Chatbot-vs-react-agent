"""
E-commerce Tools for the ReAct Agent Lab.

Scenario: "Smart E-commerce Assistant"
These tools simulate a real inventory/discount/shipping system.
Each tool has a precise description so the LLM knows exactly how to call it.
"""

from typing import Dict, Any

# ---------------------------------------------------------------------------
# Mock data (simulating a database / external API)
# ---------------------------------------------------------------------------

_STOCK_DB: Dict[str, int] = {
    "iphone": 15,
    "iphone 15": 15,
    "iphone 14": 8,
    "samsung galaxy": 22,
    "macbook pro": 5,
    "macbook air": 10,
    "airpods": 50,
    "ipad": 12,
    "laptop": 7,
}

_PRICE_DB: Dict[str, float] = {
    "iphone": 999.0,
    "iphone 15": 999.0,
    "iphone 14": 799.0,
    "samsung galaxy": 849.0,
    "macbook pro": 1999.0,
    "macbook air": 1299.0,
    "airpods": 179.0,
    "ipad": 599.0,
    "laptop": 899.0,
}

_DISCOUNT_DB: Dict[str, float] = {
    "WINNER": 10.0,   # 10% off
    "SAVE20": 20.0,   # 20% off
    "VIP50": 50.0,    # 50% off
    "STUDENT": 15.0,  # 15% off
}

_SHIPPING_RATES: Dict[str, float] = {
    # City → base rate (USD) per kg
    "hanoi": 5.0,
    "hcmc": 4.5,
    "ho chi minh": 4.5,
    "da nang": 6.0,
    "new york": 25.0,
    "london": 30.0,
    "tokyo": 28.0,
}


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def check_stock(item_name: str) -> str:
    """
    Check how many units of an item are in stock.

    Args:
        item_name: The product name (case-insensitive).

    Returns:
        A string describing the available quantity, or a not-found message.

    Example:
        check_stock("iphone") -> "iphone has 15 units in stock."
    """
    key = item_name.strip().lower()
    quantity = _STOCK_DB.get(key)

    if quantity is None:
        return f"Product '{item_name}' was not found in inventory."
    if quantity == 0:
        return f"Sorry, '{item_name}' is currently out of stock."
    return f"'{item_name}' has {quantity} units available in stock."


def get_product_price(item_name: str) -> str:
    """
    Get the current unit price (USD) of a product.

    Args:
        item_name: The product name (case-insensitive).

    Returns:
        A string with the USD price, or a not-found message.

    Example:
        get_product_price("iphone") -> "The price of iphone is $999.00 USD."
    """
    key = item_name.strip().lower()
    price = _PRICE_DB.get(key)

    if price is None:
        return f"Price for '{item_name}' not found."
    return f"The unit price of '{item_name}' is ${price:.2f} USD."


def get_discount(coupon_code: str) -> str:
    """
    Look up the discount percentage for a coupon code.

    Args:
        coupon_code: The coupon/promo code string (case-insensitive).

    Returns:
        A string describing the discount percentage, or invalid-code message.

    Example:
        get_discount("WINNER") -> "Coupon 'WINNER' gives a 10.0% discount."
    """
    key = coupon_code.strip().upper()
    discount = _DISCOUNT_DB.get(key)

    if discount is None:
        return f"Coupon code '{coupon_code}' is invalid or expired."
    return f"Coupon '{coupon_code}' gives a {discount}% discount."


def calc_shipping(weight_kg: float, destination: str) -> str:
    """
    Calculate the shipping cost (USD) based on weight and destination city.

    Args:
        weight_kg: Package weight in kilograms (positive float).
        destination: Destination city name (case-insensitive).

    Returns:
        A string with the shipping cost in USD, or a not-found message.

    Example:
        calc_shipping(1.5, "hanoi") -> "Shipping 1.5kg to Hanoi costs $7.50 USD."
    """
    city_key = destination.strip().lower()
    rate = _SHIPPING_RATES.get(city_key)

    if rate is None:
        available = ", ".join(
            c.title() for c in _SHIPPING_RATES.keys()
        )
        return (
            f"Shipping destination '{destination}' is not supported. "
            f"Available cities: {available}."
        )
    try:
        weight = float(weight_kg)
        if weight <= 0:
            return "Weight must be a positive number."
    except (TypeError, ValueError):
        return f"Invalid weight value: '{weight_kg}'. Please provide a number."

    cost = weight * rate
    return f"Shipping {weight}kg to {destination.title()} costs ${cost:.2f} USD."


# ---------------------------------------------------------------------------
# Tool Registry — used by ReActAgent to discover and execute tools
# Each entry: name, description (shown to LLM), function (called at runtime)
# ---------------------------------------------------------------------------

TOOL_REGISTRY = [
    {
        "name": "check_stock",
        "description": (
            "Check how many units of a product are available in inventory. "
            "Input: item_name (string, e.g. 'iphone'). "
            "Output: stock count or out-of-stock message."
        ),
        "function": check_stock,
    },
    {
        "name": "get_product_price",
        "description": (
            "Get the current unit price (USD) of a product. "
            "Input: item_name (string, e.g. 'iphone'). "
            "Output: price in USD."
        ),
        "function": get_product_price,
    },
    {
        "name": "get_discount",
        "description": (
            "Look up the discount percentage for a coupon/promo code. "
            "Input: coupon_code (string, e.g. 'WINNER'). "
            "Output: discount percentage or invalid-code message."
        ),
        "function": get_discount,
    },
    {
        "name": "calc_shipping",
        "description": (
            "Calculate the shipping cost based on weight (kg) and destination city. "
            "Input: weight_kg (float), destination (string, e.g. 'Hanoi'). "
            "Supported cities: Hanoi, HCMC, Da Nang, New York, London, Tokyo. "
            "Output: shipping cost in USD."
        ),
        "function": calc_shipping,
    },
]
