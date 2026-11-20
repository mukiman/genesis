"""Safe decimal arithmetic calculator tool (`calculator.py`).

Evaluates pure arithmetic expressions with decimal numbers and parentheses
using Python's `ast` module and `Decimal`, while forbidding arbitrary code
execution, attribute access, exponentiation bombs, or file I/O.
"""

from __future__ import annotations

import ast
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext
from typing import Any

MAX_EXPRESSION_LENGTH: int = 500
MAX_AST_DEPTH: int = 32
MAX_AST_NODES: int = 128
DECIMAL_PRECISION: int = 28


class CalculatorError(ValueError):
    """Controlled error raised when a calculator expression is invalid or unsafe."""


def _format_decimal(value: Decimal) -> str:
    """Format a Decimal without scientific notation and strip redundant trailing zeros after decimal point."""
    if not value.is_finite():
        raise CalculatorError("Calculation result is not a finite number.")
    # Normalize -0 to 0
    if value == 0:
        return "0"
    formatted = format(value, "f")
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted or "0"


def _eval_ast_node(node: ast.AST, depth: int = 0) -> Decimal:
    if depth > MAX_AST_DEPTH:
        raise CalculatorError(f"Expression exceeds maximum nesting depth of {MAX_AST_DEPTH}.")

    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body, depth + 1)

    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise CalculatorError(f"Unsupported literal type in calculator: {type(val).__name__}")
        try:
            dec = Decimal(str(val))
        except InvalidOperation as exc:
            raise CalculatorError(f"Invalid numeric literal: {val}") from exc
        if not dec.is_finite():
            raise CalculatorError("Non-finite numeric literals are not allowed.")
        return dec

    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand, depth + 1)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise CalculatorError(f"Unsupported unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left, depth + 1)
        right = _eval_ast_node(node.right, depth + 1)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise CalculatorError("Division by zero is not allowed.")
            try:
                return left / right
            except (DivisionByZero, InvalidOperation) as exc:
                raise CalculatorError("Division error in expression.") from exc

        raise CalculatorError(f"Unsupported binary operator: {type(node.op).__name__}")

    raise CalculatorError(f"Disallowed syntax element in calculator expression: {type(node).__name__}")


def safe_calculate(expression: str) -> dict[str, Any]:
    """Safely evaluate a pure decimal arithmetic expression and return `{'expression': ..., 'result': ...}`."""
    if not isinstance(expression, str):
        raise CalculatorError("Expression must be a string.")

    stripped = expression.strip()
    if not stripped:
        raise CalculatorError("Expression cannot be empty.")

    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError(
            f"Expression length ({len(expression)}) exceeds maximum limit of {MAX_EXPRESSION_LENGTH} characters."
        )

    try:
        parsed = ast.parse(stripped, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Syntax error in arithmetic expression: {exc.msg}") from exc

    node_count = sum(1 for _ in ast.walk(parsed))
    if node_count > MAX_AST_NODES:
        raise CalculatorError(f"Expression is too complex ({node_count} AST nodes > {MAX_AST_NODES}).")

    with localcontext() as ctx:
        ctx.prec = DECIMAL_PRECISION
        result_dec = _eval_ast_node(parsed, depth=0)
        formatted = _format_decimal(result_dec)

    return {
        "expression": expression,
        "result": formatted,
    }


def evaluate_expression(expression: str) -> str:
    """Safely evaluate an arithmetic expression and return a decimal string result."""
    return str(safe_calculate(expression)["result"])


__all__ = [
    "MAX_EXPRESSION_LENGTH",
    "MAX_AST_DEPTH",
    "CalculatorError",
    "safe_calculate",
    "evaluate_expression",
]
