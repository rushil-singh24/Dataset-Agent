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
    pass


def validate_code(code: str, columns: list[str]) -> None:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.Lambda)):
            raise SafetyError(f"Forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise SafetyError(f"Forbidden name: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRS:
                raise SafetyError(f"Forbidden attribute: {node.attr}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "__" in node.value:
                raise SafetyError("Dunder strings are not allowed in generated code.")

    referenced_columns = _find_literal_column_references(tree)
    missing = sorted(referenced_columns.difference(columns))
    if missing:
        raise SafetyError(f"Generated code references missing columns: {', '.join(missing)}")


def _find_literal_column_references(tree: ast.AST) -> set[str]:
    refs: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "df"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            refs.add(node.slice.value)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "df" and isinstance(node.slice, ast.List):
            values = [element.value for element in node.slice.elts if isinstance(element, ast.Constant) and isinstance(element.value, str)]
            refs.update(values)
    return refs


def execute_analysis(code: str, df: pd.DataFrame, chart_type: str = "none") -> tuple[Any, list[TableResult], list[ChartResult]]:
    validate_code(code, [str(column) for column in df.columns])
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_execute, code, df.copy(), chart_type)
        try:
            return future.result(timeout=8)
        except TimeoutError as exc:
            raise SafetyError("Analysis timed out.") from exc


def _execute(code: str, df: pd.DataFrame, chart_type: str) -> tuple[Any, list[TableResult], list[ChartResult]]:
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
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "range": range,
            "round": round,
            "str": str,
            "sum": sum,
        },
    }
    exec(compile(code, "<generated_analysis>", "exec"), exec_env, exec_env)
    result = exec_env.get("result")
    tables = [_to_table(result)] if result is not None else []
    charts = _create_chart(result, chart_type) if chart_type != "none" else []
    return _json_ready(result), tables, charts


def _to_table(result: Any) -> TableResult:
    if isinstance(result, pd.Series):
        frame = result.reset_index()
    elif isinstance(result, pd.DataFrame):
        frame = result
    elif isinstance(result, dict):
        frame = pd.DataFrame([{key: _table_cell(value) for key, value in result.items()}])
    else:
        frame = pd.DataFrame([{"result": result}])
    frame = frame.head(100).copy()
    return TableResult(
        title="Analysis Result",
        columns=[str(column) for column in frame.columns],
        rows=frame.where(pd.notna(frame), None).to_dict(orient="records"),
    )


def _table_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return value


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
    columns = list(frame.columns)
    numeric = list(frame.select_dtypes(include="number").columns)

    if chart_type == "histogram" and numeric:
        frame[numeric[0]].plot(kind="hist", ax=ax, bins=30)
        ax.set_xlabel(str(numeric[0]))
    elif chart_type == "scatter" and len(numeric) >= 2:
        ax.scatter(frame[numeric[0]], frame[numeric[1]])
        ax.set_xlabel(str(numeric[0]))
        ax.set_ylabel(str(numeric[1]))
    elif chart_type == "heatmap" and len(numeric) >= 2:
        corr = frame[numeric].corr(numeric_only=True)
        image = ax.imshow(corr, cmap="viridis")
        ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)), corr.index)
        fig.colorbar(image, ax=ax)
    elif numeric:
        x_column = columns[0]
        y_column = numeric[-1]
        if chart_type == "line":
            ax.plot(frame[x_column].astype(str), frame[y_column])
        else:
            ax.bar(frame[x_column].astype(str), frame[y_column])
        ax.set_xlabel(str(x_column))
        ax.set_ylabel(str(y_column))
        ax.tick_params(axis="x", labelrotation=35)
    else:
        plt.close(fig)
        return []

    ax.set_title("Generated Visualization")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return [ChartResult(artifact_id=artifact_id, title="Generated Visualization", type=chart_type, url=f"/api/artifacts/{artifact_id}")]


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.head(50).where(pd.notna(value), None).to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.head(50).where(pd.notna(value), None).to_dict()
    if hasattr(value, "item"):
        return value.item()
    return value
