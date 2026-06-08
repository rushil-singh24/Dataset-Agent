"""
safe_exec.py  –  sandboxed execution of LLM-generated pandas code

Changes from v1:
- execute_analysis() now returns a richer ExecutionError on failure so the
  retry loop in main.py can pass a useful message back to the LLM.
- All other behaviour (AST validation, forbidden names, timeout) is unchanged.
"""

from __future__ import annotations

import ast
import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from .store import ARTIFACT_DIR, DATA_DIR

os.environ.setdefault("MPLCONFIGDIR", str(DATA_DIR / "mplconfig"))

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .schemas import ChartResult, TableResult


# ---------------------------------------------------------------------------
# Forbidden identifiers
# ---------------------------------------------------------------------------

FORBIDDEN_NAMES = {
    "__import__",
    "open",
    "exec",
    "eval",
    "compile",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "input",
    "help",
    "exit",
    "quit",
}

FORBIDDEN_ATTRS = {
    "to_csv",
    "to_excel",
    "to_parquet",
    "to_pickle",
    "read_csv",
    "read_excel",
    "read_parquet",
    "read_pickle",
    "system",
    "popen",
    "remove",
    "unlink",
    "rmdir",
    "mkdir",
    "write",
}


class SafetyError(ValueError):
    """Raised when generated code fails the static safety check."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def execute_analysis(
    code: str,
    df: pd.DataFrame,
    chart_type: str = "none",
) -> tuple[Any, list[TableResult], list[ChartResult]]:
    """
    Validate `code` statically then execute it in a sandboxed environment.

    Returns (raw_result, tables, charts).
    Raises SafetyError  – static validation failed (bad syntax, forbidden names, etc.)
    Raises RuntimeError – code ran but raised an exception at runtime.
                          The message includes the original exception for the LLM retry.
    Raises TimeoutError equivalent wrapped in RuntimeError – execution timed out.
    """
    validate_code(code, [str(col) for col in df.columns])

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_execute, code, df.copy(), chart_type)
        try:
            return future.result(timeout=15)
        except TimeoutError as exc:
            raise RuntimeError(
                "Code execution timed out after 15 seconds. "
                "Try a simpler or more targeted query."
            ) from exc
        except SafetyError:
            raise
        except Exception as exc:
            # Wrap with a descriptive message for the LLM retry
            raise RuntimeError(
                f"{type(exc).__name__}: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# AST validator
# ---------------------------------------------------------------------------

def validate_code(code: str, columns: list[str]) -> None:
    """
    Static safety check.  Raises SafetyError if the code uses forbidden
    constructs or references columns that don't exist in the dataframe.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SafetyError(f"Syntax error in generated code: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.Global,
                ast.Nonlocal,
                ast.With,
                ast.AsyncWith,
                ast.Lambda,
            ),
        ):
            raise SafetyError(f"Forbidden syntax node: {type(node).__name__}")

        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise SafetyError(f"Forbidden built-in: {node.id!r}")

        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRS:
                raise SafetyError(f"Forbidden attribute: {node.attr!r}")

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "__" in node.value:
                raise SafetyError("Dunder strings are not permitted in generated code.")

    # Check that any literal column subscripts actually exist
    referenced = _find_literal_column_references(tree)
    missing = sorted(referenced - set(columns))
    if missing:
        raise SafetyError(
            f"Generated code references columns that don't exist in the dataset: "
            f"{', '.join(missing)}.  Available columns: {', '.join(columns[:20])}"
        )


def _find_literal_column_references(tree: ast.AST) -> set[str]:
    refs: set[str] = set()
    for node in ast.walk(tree):
        # df['col']
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "df"
        ):
            if isinstance(node.slice, ast.Constant) and isinstance(
                node.slice.value, str
            ):
                refs.add(node.slice.value)
            # df[['col1', 'col2']]
            if isinstance(node.slice, ast.List):
                for elt in node.slice.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        refs.add(elt.value)
    return refs


# ---------------------------------------------------------------------------
# Execution sandbox
# ---------------------------------------------------------------------------

def _execute(
    code: str, df: pd.DataFrame, chart_type: str
) -> tuple[Any, list[TableResult], list[ChartResult]]:
    exec_env: dict[str, Any] = {
        "df": df,
        "pd": pd,
        "np": np,
        "duckdb": duckdb,
        "__builtins__": {
            "__import__": __import__,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        },
    }

    exec(compile(code, "<generated_analysis>", "exec"), exec_env, exec_env)  # noqa: S102

    result = exec_env.get("result")
    tables = [_to_table(result)] if result is not None else []
    charts = _create_chart(result, chart_type) if chart_type != "none" else []
    return _json_ready(result), tables, charts


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def _to_table(result: Any) -> TableResult:
    if isinstance(result, pd.Series):
        frame = result.reset_index()
    elif isinstance(result, pd.DataFrame):
        frame = result
    elif isinstance(result, dict):
        # Flatten nested dicts/lists so they display cleanly
        frame = pd.DataFrame(
            [{k: _table_cell(v) for k, v in result.items()}]
        )
    elif isinstance(result, list) and result and isinstance(result[0], dict):
        frame = pd.DataFrame(result)
    else:
        frame = pd.DataFrame([{"result": result}])

    frame = frame.head(100).copy()
    return TableResult(
        title="Analysis Result",
        columns=[str(c) for c in frame.columns],
        rows=frame.where(pd.notna(frame), None).to_dict(orient="records"),
    )


def _table_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return value


# ---------------------------------------------------------------------------
# Chart builder  (unchanged logic, just cleaner structure)
# ---------------------------------------------------------------------------

def _create_chart(result: Any, chart_type: str) -> list[ChartResult]:
    if result is None:
        return []

    frame = result.reset_index() if isinstance(result, pd.Series) else result
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_id = f"{uuid4().hex}.png"
    path = ARTIFACT_DIR / artifact_id

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    cols = list(frame.columns)
    numeric = list(frame.select_dtypes(include="number").columns)

    try:
        if chart_type == "histogram" and numeric:
            frame[numeric[0]].plot(kind="hist", ax=ax, bins=30)
            ax.set_xlabel(str(numeric[0]))

        elif chart_type == "scatter" and len(numeric) >= 2:
            ax.scatter(frame[numeric[0]], frame[numeric[1]])
            ax.set_xlabel(str(numeric[0]))
            ax.set_ylabel(str(numeric[1]))

        elif chart_type == "heatmap" and len(numeric) >= 2:
            corr = frame[numeric].corr(numeric_only=True)
            img = ax.imshow(corr, cmap="viridis")
            ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(corr.index)), corr.index)
            fig.colorbar(img, ax=ax)

        elif numeric:
            x_col = cols[0]
            y_col = numeric[-1]
            if chart_type == "line":
                ax.plot(frame[x_col].astype(str), frame[y_col])
            else:
                ax.bar(frame[x_col].astype(str), frame[y_col])
            ax.set_xlabel(str(x_col))
            ax.set_ylabel(str(y_col))
            ax.tick_params(axis="x", labelrotation=35)

        else:
            plt.close(fig)
            return []

        ax.set_title("Generated Visualization")
        fig.savefig(path, dpi=140)
    finally:
        plt.close(fig)

    return [
        ChartResult(
            artifact_id=artifact_id,
            title="Generated Visualization",
            type=chart_type,
            url=f"/api/artifacts/{artifact_id}",
        )
    ]


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.head(50).where(pd.notna(value), None).to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.head(50).where(pd.notna(value), None).to_dict()
    if hasattr(value, "item"):
        return value.item()
    return value