"""LLM-first analysis agent with a generic deterministic fallback."""

from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any

import httpx
from pydantic import ValidationError

from .schemas import AnalysisPlan, Confidence, GeneratedAnalysis, SummaryResponse

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", "3"))
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "90"))


async def build_analysis(
    message: str,
    dataset_context: str,
    columns: list[str],
    history: list[dict[str, Any]],
) -> GeneratedAnalysis:
    last_error: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        prompt = _build_analysis_prompt(
            message=message,
            dataset_context=dataset_context,
            columns=columns,
            history=history,
            previous_error=last_error,
            attempt=attempt,
        )
        try:
            raw = await _ollama_json(prompt)
            return _parse_analysis(raw)
        except _OllamaError:
            return _fallback_analysis(message, columns, ollama_unavailable=True)
        except _BadResponseError as exc:
            last_error = str(exc)
            if attempt == MAX_RETRIES:
                return _fallback_analysis(message, columns, ollama_unavailable=False)

    return _fallback_analysis(message, columns, ollama_unavailable=False)


async def build_analysis_with_code_retry(
    message: str,
    dataset_context: str,
    columns: list[str],
    history: list[dict[str, Any]],
    execution_error: str,
    previous_code: str,
) -> GeneratedAnalysis:
    prompt = _build_code_fix_prompt(
        message=message,
        dataset_context=dataset_context,
        columns=columns,
        history=history,
        bad_code=previous_code,
        error=execution_error,
    )
    try:
        raw = await _ollama_json(prompt)
        return _parse_analysis(raw)
    except (_OllamaError, _BadResponseError):
        return _fallback_analysis(message, columns, ollama_unavailable=True)


async def summarize_result(
    message: str,
    plan: AnalysisPlan,
    result: Any,
    educational_mode: bool,
    missing_relevant_values: bool,
) -> SummaryResponse:
    result_text = _serialise_result(result)

    extra = ""
    if missing_relevant_values:
        extra += "\nSome columns relevant to this question contain missing values. Mention this caveat."
    if educational_mode:
        extra += "\nEducational Mode is enabled. Briefly explain the analysis method used."

    prompt = textwrap.dedent(f"""
        You are DataChat AI, a careful data analyst explaining a computed pandas result.
        Respond with ONLY a JSON object. Do not use markdown outside the JSON.

        Requirements:
        - Answer the user's question directly first.
        - Use exact numbers from the execution result when available.
        - Do not invent dataset facts that are not shown in the result.
        - If the result is empty, say that no matching rows or values were found.
        - Keep the answer concise and easy to understand.

        Original question:
        {message}

        Analysis plan:
        - Category: {plan.category}
        - Operations performed: {", ".join(plan.planned_operations)}
        - Columns used: {", ".join(plan.selected_columns) or "none specified"}

        Execution result:
        {result_text}
        {extra}

        Return this JSON shape:
        {{
          "answer": "A clear, direct answer in plain English. Reference specific numbers from the result.",
          "confidence": {{
            "level": "High|Medium|Low",
            "reason": "One sentence explaining confidence level."
          }}
        }}
    """).strip()

    try:
        raw = await _ollama_json(prompt)
        return SummaryResponse.model_validate(raw)
    except (_OllamaError, ValidationError, _BadResponseError):
        prefix = "Ollama is unavailable, using limited fallback mode. " if _is_fallback_plan(plan) else ""
        return SummaryResponse(
            answer=prefix + _fallback_summary_from_result(result),
            confidence=Confidence(level="Medium", reason="The numeric result was computed successfully, but natural-language summary generation was unavailable."),
        )


def _build_analysis_prompt(
    *,
    message: str,
    dataset_context: str,
    columns: list[str],
    history: list[dict[str, Any]],
    previous_error: str | None,
    attempt: int,
) -> str:
    history_block = _format_history(history)
    error_block = ""
    if previous_error:
        error_block = textwrap.dedent(f"""
            IMPORTANT: Your previous attempt #{attempt - 1} failed:
            {previous_error}
            Fix the issue. Do not repeat the same mistake.
        """).strip()

    return textwrap.dedent(f"""
        You are DataChat AI, a careful expert data analyst agent.
        You have access to a pandas DataFrame named `df`.

        Main objective:
        Answer the user's question accurately using ONLY the uploaded dataset.

        Before writing code, reason about grounding:
        - Match the user's words to the exact column names in COLUMNS.
        - If the user asks for a value/filter/category, use case-insensitive matching where appropriate.
        - If no relevant column exists, mark grounding_status as "Not Found".
        - If the request cannot be answered from the dataset, classify it as "Unsupported Request".
        - Do not assume domain meaning beyond column names, sample values, and statistics.

        CODE RULES:
        - The DataFrame is already loaded as `df`. Do NOT import anything.
        - Do NOT use open, exec, eval, import, __import__, os, sys, subprocess, duckdb, file paths, or network calls.
        - Do NOT write files.
        - Assign final output to a variable named `result`.
        - `result` should be a DataFrame, Series, dict, scalar, or list.
        - Use pandas/numpy operations only.
        - Use numeric_only=True for broad numeric aggregations where appropriate.
        - Handle NaN values gracefully.
        - Only reference columns from the provided COLUMNS list.
        - Prefer small, readable results over huge outputs. Use head(50) for row lists unless the user asks otherwise.
        - For groupby aggregations, include counts when useful so the answer has context.
        - For yes/no questions, compute evidence and return it, not just True/False.

        DATASET CONTEXT:
        {dataset_context}

        COLUMNS:
        {json.dumps(columns)}

        CONVERSATION HISTORY:
        {history_block}

        {error_block}

        USER QUESTION:
        {message}

        Respond with ONLY this JSON object:
        {{
          "classification": "<one of: Dataset Information | Statistical Analysis | Correlation Analysis | Visualization Request | Data Quality Inspection | Trend Analysis | Aggregation Request | Unsupported Request>",
          "plan": {{
            "category": "<same as classification>",
            "question_understanding": "<one sentence>",
            "selected_columns": ["<only exact real column names from COLUMNS>"],
            "column_selection_reason": "<why these columns answer the question, or why none were found>",
            "planned_operations": ["<step 1>", "<step 2>"],
            "output_type": "<Text | Table | Chart | Table + Chart | Text + Table>",
            "estimated_complexity": "<Low | Medium | High>",
            "grounding_status": "<Ready | Needs Clarification | Not Found | Unsupported>",
            "status": "<Ready to Execute | Needs Clarification | No Execution Needed>"
          }},
          "code": "<valid pandas code string, or null if unsupported>",
          "chart_type": "<line | bar | scatter | histogram | heatmap | none>"
        }}
    """).strip()


def _build_code_fix_prompt(
    *,
    message: str,
    dataset_context: str,
    columns: list[str],
    history: list[dict[str, Any]],
    bad_code: str,
    error: str,
) -> str:
    return textwrap.dedent(f"""
        Your previous pandas code failed. Fix it.
        Return ONLY the required JSON object.

        CODE RULES:
        - DataFrame is `df`. Do NOT import anything.
        - Do NOT use open, exec, eval, import, __import__, os, sys, subprocess, duckdb, file paths, or network calls.
        - Do NOT write files.
        - Assign final output to `result`.
        - Only reference exact columns from COLUMNS.
        - Keep the code simple and robust to missing values.

        DATASET CONTEXT:
        {dataset_context}

        COLUMNS:
        {json.dumps(columns)}

        CONVERSATION HISTORY:
        {_format_history(history)}

        ORIGINAL QUESTION:
        {message}

        FAILED CODE:
        {bad_code}

        ERROR:
        {error}

        Respond with ONLY the same JSON shape as before.
    """).strip()


def _fallback_analysis(message: str, columns: list[str], *, ollama_unavailable: bool) -> GeneratedAnalysis:
    lower = message.lower()
    selected = _select_columns(lower, columns)
    prefix = "Ollama is unavailable, using limited fallback mode. " if ollama_unavailable else "Using limited fallback mode. "

    if _asks_schema(lower):
        return _analysis(
            message,
            "Dataset Information",
            [],
            prefix + "All columns are relevant for schema inspection.",
            ["Inspect column names", "Inspect dtypes", "Count missing and unique values"],
            "Table + Text",
            "Ready",
            "result = pd.DataFrame({'column': df.columns, 'dtype': df.dtypes.astype(str).values, 'missing_count': df.isna().sum().values, 'unique_count': df.nunique(dropna=True).values})",
        )

    if _asks_row_count(lower):
        return _analysis(
            message,
            "Dataset Information",
            [],
            prefix + "Row and column counts are available directly from the dataframe shape.",
            ["Count rows", "Count columns"],
            "Text + Table",
            "Ready",
            "result = {'row_count': int(len(df)), 'column_count': int(len(df.columns))}",
        )

    if _asks_missing(lower):
        return _analysis(
            message,
            "Data Quality Inspection",
            [],
            prefix + "All columns are checked for missing values.",
            ["Calculate missing counts", "Calculate missing percentages"],
            "Table + Text",
            "Ready",
            "result = pd.DataFrame({'column': df.columns, 'missing_count': df.isna().sum().values, 'missing_percent': (df.isna().mean().values * 100).round(2)})",
        )

    if "correl" in lower:
        return _analysis(
            message,
            "Correlation Analysis",
            [],
            prefix + "Correlation fallback uses all numeric columns.",
            ["Select numeric columns", "Compute correlation matrix"],
            "Table + Text",
            "Ready",
            "numeric = df.select_dtypes(include='number'); result = numeric.corr(numeric_only=True).round(3) if len(numeric.columns) > 1 else pd.DataFrame({'message': ['At least two numeric columns are needed for correlation.']})",
        )

    if _asks_average(lower) and selected:
        col = selected[-1]
        return _analysis(
            message,
            "Statistical Analysis",
            [col],
            prefix + f"The question appears to ask for a basic numeric summary of {col}.",
            [f"Convert {col} to numeric where possible", "Calculate descriptive statistics"],
            "Table + Text",
            "Ready",
            f"series = pd.to_numeric(df[{col!r}], errors='coerce'); result = series.describe().round(4)",
        )

    if _asks_value_counts(lower):
        col = selected[-1] if selected else None
        if col:
            code = f"counts = df[{col!r}].dropna().astype(str).value_counts().head(25); result = pd.DataFrame({{'value': counts.index, 'count': counts.values, 'percent': (counts.values / max(len(df), 1) * 100).round(2)}})"
            selected_cols = [col]
            reason = prefix + f"Value counts are computed for {col}."
        else:
            code = "rows = []\nfor column in df.columns:\n    series = df[column].dropna().astype(str)\n    if series.nunique(dropna=True) <= 25:\n        counts = series.value_counts().head(10)\n        for value, count in counts.items():\n            rows.append({'column': str(column), 'value': str(value), 'count': int(count), 'percent': round(float(count) / max(len(df), 1) * 100, 2)})\nresult = pd.DataFrame(rows)"
            selected_cols = []
            reason = prefix + "No specific column was detected, so low-cardinality columns are summarized."
        return _analysis(message, "Statistical Analysis", selected_cols, reason, ["Compute value counts", "Compute percentages"], "Table + Text", "Ready", code)

    if _asks_sample(lower):
        limit = _extract_limit(lower)
        return _analysis(
            message,
            "Dataset Information",
            [],
            prefix + "Sample rows can be shown directly from the uploaded dataframe.",
            [f"Return first {limit} rows"],
            "Table + Text",
            "Ready",
            f"result = df.head({limit})",
        )

    return _analysis(
        message,
        "Unsupported Request",
        [],
        prefix + "The limited fallback only supports schema, row/column count, missing values, descriptive stats, value counts, correlations, and sample rows. Make sure Ollama is running for richer analysis.",
        ["Refuse unsupported fallback request"],
        "Text",
        "Unsupported",
        None,
    )


def _analysis(
    message: str,
    category: str,
    selected_columns: list[str],
    reason: str,
    operations: list[str],
    output_type: str,
    grounding: str,
    code: str | None,
) -> GeneratedAnalysis:
    plan = AnalysisPlan(
        category=category,  # type: ignore[arg-type]
        question_understanding=message,
        selected_columns=selected_columns,
        column_selection_reason=reason,
        planned_operations=operations,
        output_type=output_type,
        estimated_complexity="Low",
        grounding_status=grounding,  # type: ignore[arg-type]
        status="Ready to Execute" if code else "No Execution Needed",
    )
    return GeneratedAnalysis(classification=category, plan=plan, code=code, chart_type="none")  # type: ignore[arg-type]


def _select_columns(lower_message: str, columns: list[str]) -> list[str]:
    selected: list[str] = []
    normalized_message = _normalize(lower_message)
    for col in columns:
        normalized_col = _normalize(str(col))
        if normalized_col and normalized_col in normalized_message:
            selected.append(str(col))
    if not selected:
        for col in columns:
            tokens = [token for token in _normalize(str(col)).split() if len(token) > 2]
            if tokens and all(token in normalized_message for token in tokens):
                selected.append(str(col))
    return selected[:6]


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _asks_schema(lower: str) -> bool:
    return any(term in lower for term in ["schema", "columns", "column names", "dtypes", "data types", "fields"])


def _asks_row_count(lower: str) -> bool:
    return any(term in lower for term in ["how many rows", "row count", "number of rows", "how many records", "records", "shape", "dimensions"])


def _asks_missing(lower: str) -> bool:
    return any(term in lower for term in ["missing", "null", "nan", "blank", "quality", "complete", "completeness"])


def _asks_average(lower: str) -> bool:
    return any(term in lower for term in ["average", "mean", "median", "minimum", "maximum", "min", "max", "describe", "summary stats", "statistics"])


def _asks_value_counts(lower: str) -> bool:
    return any(term in lower for term in ["value counts", "distribution", "breakdown", "frequency", "most common", "categories", "percent", "percentage"])


def _asks_sample(lower: str) -> bool:
    return any(term in lower for term in ["sample", "examples", "show rows", "first rows", "preview", "head"])


def _extract_limit(lower: str) -> int:
    match = re.search(r"\b(\d{1,2})\b", lower)
    if match:
        return max(1, min(int(match.group(1)), 25))
    return 10


def _is_fallback_plan(plan: AnalysisPlan) -> bool:
    return "fallback" in plan.column_selection_reason.lower() or "ollama is unavailable" in plan.column_selection_reason.lower()


def _format_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "(no previous conversation)"
    lines = []
    for item in history[-6:]:
        if isinstance(item, dict):
            role = item.get("role", "unknown")
            content = item.get("content", "")
            if content:
                lines.append(f"{role.upper()}: {str(content)[:300]}")
    return "\n".join(lines) if lines else "(no previous conversation)"


def _serialise_result(result: Any) -> str:
    try:
        import pandas as pd  # noqa: PLC0415

        if isinstance(result, pd.DataFrame):
            return result.head(50).to_string(index=True)
        if isinstance(result, pd.Series):
            return result.head(50).to_string()
    except ImportError:
        pass

    try:
        return json.dumps(result, default=str)[:4000]
    except Exception:
        return str(result)[:4000]


def _fallback_summary_from_result(result: Any) -> str:
    """Create a useful plain-English answer when Ollama summary generation fails."""
    try:
        import pandas as pd  # noqa: PLC0415

        if isinstance(result, pd.DataFrame):
            if result.empty:
                return "The analysis ran successfully, but the result table is empty."
            rows, cols = result.shape
            if rows == 1 and cols <= 8:
                first = result.iloc[0].to_dict()
                parts = [f"{key}: {_short_value(value)}" for key, value in first.items()]
                return "Result: " + ", ".join(parts) + "."
            preview = result.head(3).to_dict(orient="records")
            return f"I computed a result table with {rows:,} rows and {cols:,} columns. First rows: {_short_value(preview)}."

        if isinstance(result, pd.Series):
            if result.empty:
                return "The analysis ran successfully, but there were no values to summarize."
            preview = result.head(8).to_dict()
            return f"Result: {_short_value(preview)}."
    except ImportError:
        pass

    if isinstance(result, dict):
        parts = [f"{key}: {_short_value(value)}" for key, value in list(result.items())[:8]]
        return "Result: " + ", ".join(parts) + "."
    if isinstance(result, list):
        return f"Result: {_short_value(result[:8])}."
    if result is None:
        return "The analysis completed, but no result was returned."
    return f"Result: {_short_value(result)}."


def _short_value(value: Any, limit: int = 700) -> str:
    text = str(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _parse_analysis(raw: dict[str, Any]) -> GeneratedAnalysis:
    try:
        plan_raw = raw.get("plan", {})
        if isinstance(plan_raw.get("selected_columns"), str):
            plan_raw["selected_columns"] = [plan_raw["selected_columns"]]
        if isinstance(plan_raw.get("planned_operations"), str):
            plan_raw["planned_operations"] = [plan_raw["planned_operations"]]

        valid_classifications = {
            "Dataset Information",
            "Statistical Analysis",
            "Correlation Analysis",
            "Visualization Request",
            "Data Quality Inspection",
            "Trend Analysis",
            "Aggregation Request",
            "Unsupported Request",
        }
        classification = raw.get("classification", "Statistical Analysis")
        if classification not in valid_classifications:
            classification = "Statistical Analysis"
        raw["classification"] = classification
        plan_raw["category"] = classification

        valid_grounding = {"Ready", "Needs Clarification", "Not Found", "Unsupported"}
        if plan_raw.get("grounding_status", "Ready") not in valid_grounding:
            plan_raw["grounding_status"] = "Ready"

        real_columns = set(raw.get("available_columns", [])) or None
        if real_columns:
            plan_raw["selected_columns"] = [col for col in plan_raw.get("selected_columns", []) if col in real_columns]

        valid_charts = {"line", "bar", "scatter", "histogram", "heatmap", "none"}
        if raw.get("chart_type", "none") not in valid_charts:
            raw["chart_type"] = "none"

        raw["plan"] = plan_raw
        return GeneratedAnalysis.model_validate(raw)
    except (ValidationError, KeyError, TypeError) as exc:
        raise _BadResponseError(f"Could not parse model response: {exc}") from exc


async def _ollama_json(prompt: str) -> dict[str, Any]:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        raise _OllamaError(str(exc)) from exc

    text = data.get("response", "")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise _BadResponseError(f"Model did not return valid JSON: {text[:500]}") from exc


class _OllamaError(RuntimeError):
    pass


class _BadResponseError(RuntimeError):
    pass
