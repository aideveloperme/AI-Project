"""Deterministic tools and tasks for the agent-latency benchmark.

The tools are pure functions with canned data so that runs are reproducible
and tool time can be separated from model time. ``search_docs`` returns a
few hundred tokens on purpose: real tool outputs (RAG chunks, API JSON) are
what make agent contexts grow step after step.
"""

from __future__ import annotations

import ast
import json
import operator
from dataclasses import dataclass
from typing import Any

ORDERS: dict[str, dict[str, Any]] = {
    "A-1042": {
        "items": [{"sku": "KB-01", "qty": 1, "price": 75.00}, {"sku": "MS-07", "qty": 2, "price": 25.00}],
        "subtotal": 125.00,
        "placed_days_ago": 3,
        "status": "shipped",
    },
    "B-2077": {
        "items": [{"sku": "CB-11", "qty": 3, "price": 29.99}],
        "subtotal": 89.97,
        "placed_days_ago": 10,
        "status": "processing",
    },
    "C-3100": {
        "items": [{"sku": "MN-27", "qty": 1, "price": 349.00}],
        "subtotal": 349.00,
        "placed_days_ago": 45,
        "status": "delivered",
    },
}
SHIPPING_DAYS = {"US": 3, "EU": 6, "APAC": 9}

POLICY_DOC = (
    "Refund policy: customers may request a full refund within 30 days of the order being placed. "
    "After 30 days, orders are not eligible for a refund but may be eligible for store credit at the "
    "discretion of a support lead. Items must be returned in original packaging. Shipping fees are "
    "non-refundable unless the item arrived damaged. Sales tax is charged at 8 percent on the order "
    "subtotal for all regions covered by this policy. Orders in status 'processing' can be cancelled "
    "without a fee. Orders in status 'shipped' must be returned after delivery. "
)

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.Pow: operator.pow,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str = "") -> str:
    try:
        val = _safe_eval(ast.parse(str(expression), mode="eval"))
        return json.dumps({"result": round(val, 4)})
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as e:
        return json.dumps({"error": f"cannot evaluate: {e}"})


def lookup_order(order_id: str = "") -> str:
    o = ORDERS.get(str(order_id).strip().upper())
    return json.dumps({"order_id": order_id, **o}) if o else json.dumps({"error": f"order {order_id} not found"})


def get_shipping_eta(order_id: str = "", region: str = "US") -> str:
    if str(order_id).strip().upper() not in ORDERS:
        return json.dumps({"error": f"order {order_id} not found"})
    days = SHIPPING_DAYS.get(str(region).upper())
    if days is None:
        return json.dumps({"error": f"unknown region {region}; use one of {sorted(SHIPPING_DAYS)}"})
    return json.dumps({"order_id": order_id, "region": region, "eta_days": days})


def search_docs(query: str = "") -> str:
    # Repeat the policy to make the tool output realistically long (~400 tokens).
    return json.dumps({"query": query, "results": [{"title": "Customer refund & tax policy", "text": POLICY_DOC * 3}]})


TOOL_FUNCS = {f.__name__: f for f in (calculator, lookup_order, get_shipping_eta, search_docs)}


def _fn(name: str, desc: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


TOOL_SCHEMAS = [
    _fn(
        "calculator",
        "Evaluate an arithmetic expression, e.g. '125*1.08'.",
        {"expression": {"type": "string"}},
        ["expression"],
    ),
    _fn(
        "lookup_order",
        "Fetch an order's items, subtotal, age and status.",
        {"order_id": {"type": "string"}},
        ["order_id"],
    ),
    _fn(
        "get_shipping_eta",
        "Shipping ETA in days for an order to a region (US, EU, APAC).",
        {"order_id": {"type": "string"}, "region": {"type": "string"}},
        ["order_id", "region"],
    ),
    _fn("search_docs", "Search the internal policy knowledge base.", {"query": {"type": "string"}}, ["query"]),
]


def execute_tool(name: str, arguments: str) -> str:
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool {name}"})
    try:
        args = json.loads(arguments) if arguments else {}
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
        return fn(**{k: v for k, v in args.items() if isinstance(k, str)})
    except (ValueError, TypeError) as e:
        return json.dumps({"error": f"bad arguments: {e}"})


@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    expected: tuple[str, ...]  # any of these substrings (case-insensitive) => success


TASKS = [
    Task("order_total_tax", "What is the total for order A-1042 including sales tax? Give the amount.", ("135",)),
    Task("avg_item_price", "For order B-2077, what is the average price per item unit?", ("29.99",)),
    Task(
        "refund_eligibility",
        "Is order C-3100 eligible for a full refund under our policy? Answer yes or no and why.",
        ("not eligible", "no"),
    ),
    Task("combined_total", "What is the combined subtotal of orders A-1042 and B-2077?", ("214.97",)),
    Task("shipping_eta", "How many days will order B-2077 take to ship to the EU?", ("6",)),
]
