import json

from servebench.agent import system_prompt
from servebench.agent_tools import ORDERS, TASKS, calculator, execute_tool


def test_calculator_is_safe():
    assert json.loads(calculator("125*1.08"))["result"] == 135.0
    assert "error" in json.loads(calculator("__import__('os').system('true')"))
    assert "error" in json.loads(calculator("1/0"))


def test_execute_tool_handles_bad_input():
    assert "unknown tool" in execute_tool("rm", "{}")
    assert "bad arguments" in execute_tool("calculator", "not json")
    assert "bad arguments" in execute_tool("lookup_order", '{"nope": 1}')


def test_task_answers_match_data():
    by_id = {t.task_id: t for t in TASKS}
    assert f"{ORDERS['A-1042']['subtotal'] * 1.08:.0f}" in by_id["order_total_tax"].expected
    assert f"{ORDERS['A-1042']['subtotal'] + ORDERS['B-2077']['subtotal']:.2f}" in by_id["combined_total"].expected


def test_system_prompt_size():
    assert 1500 < len(system_prompt(2000).split()) < 2500
