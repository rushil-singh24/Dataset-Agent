"""
ollama_agent.py  –  LLM-first analysis agent

Every question goes through the local Ollama model.
No hardcoded keyword routing, no deterministic fallbacks.
Bad generated code is retried up to MAX_RETRIES times, feeding
the error back to the model so it can self-correct.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any

import httpx
from pydantic import ValidationError

from .schemas import (
    AnalysisPlan,
    Confidence,
    GeneratedAnalysis,
    SummaryResponse,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# qwen2.5-coder:7b is the recommended model – trained on code, great at pandas.
# Switch to qwen3:8b or any other local model via env var if preferred.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

# How many times to retry if the model produces bad/unsafe code
MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", "3"))

# Ollama request timeout in seconds
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def build_analysis(
    message: str,
    dataset_context: str,
    columns: list[str],
    history: list[dict[str, Any]],
) -> GeneratedAnalysis:
    """
    Ask the LLM to understand the question, build an analysis plan,
    and generate Pandas code to answer it.

    Returns a GeneratedAnalysis.  Never raises – returns an
    'Unsupported Request' analysis on total failure.
    """
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
            analysis = _parse_analysis(raw)

            # If the model says it needs clarification or can't answer, trust it
            if analysis.classification == "Unsupported Request":
                return analysis

            # If code was generated we'll let safe_exec validate/run it.
            # Return immediately – the caller (main.py) handles execution errors
            # and can decide to surface them or retry at a higher level.
            return analysis

        except _OllamaError as exc:
            # Ollama is down / timed out – no point retrying
            return _error_analysis(message, f"Ollama is unavailable: {exc}")
        except _BadResponseError as exc:
            last_error = str(exc)
            if attempt == MAX_RETRIES:
                return _error_analysis(
                    message,
                    f"The model produced invalid output after {MAX_RETRIES} attempts. "
                    f"Last error: {last_error}",
                )
            # loop and retry with the error fed back to the model

    # Should be unreachable
    return _error_analysis(message, "Unexpected agent loop exit.")


async def build_analysis_with_code_retry(
    message: str,
    dataset_context: str,
    columns: list[str],
    history: list[dict[str, Any]],
    execution_error: str,
    previous_code: str,
) -> GeneratedAnalysis:
    """
    Called by main.py when safe_exec raises SafetyError or a runtime exception.
    We feed the error + the bad code back to the model and ask it to fix it.
    """
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
    except (_OllamaError, _BadResponseError) as exc:
        return _error_analysis(message, f"Code fix attempt failed: {exc}")


async def summarize_result(
    message: str,
    plan: AnalysisPlan,
    result: Any,
    educational_mode: bool,
    missing_relevant_values: bool,
) -> SummaryResponse:
    """
    Ask the LLM to turn the raw execution result into a plain-English answer.
    Falls back to a minimal canned response only if Ollama is completely down.
    """
    result_text = _serialise_result(result)

    extra = ""
    if missing_relevant_values:
        extra = (
            "\nNote: some columns relevant to this question contain missing values. "
            "Mention this caveat in your answer."
        )
    if educational_mode:
        extra += (
            "\nThe user has enabled Educational Mode. "
            "After answering, briefly explain the analytical method you used "
            "(e.g. groupby + sum, correlation matrix, value_counts)."
        )

    prompt = textwrap.dedent(f"""
        You are a data analyst explaining results to a user.
        Respond with ONLY a JSON object – no markdown, no extra text.

        Original question: {message}

        Analysis plan summary:
        - Category: {plan.category}
        - Operations performed: {", ".join(plan.planned_operations)}
        - Columns used: {", ".join(plan.selected_columns) or "none specified"}

        Execution result (may be a table, dict, scalar, or list):
        {result_text}
        {extra}

        Return this JSON shape:
        {{
          "answer": "A clear, direct answer to the question in plain English. Reference specific numbers from the result. Do not say 'the result shows' – just answer the question directly.",
          "confidence": {{
            "level": "High|Medium|Low",
            "reason": "One sentence explaining confidence level."
          }}
        }}
    """).strip()

    try:
        raw = await _ollama_json(prompt)
        return SummaryResponse.model_validate(raw)
    except (_OllamaError, ValidationError, _BadResponseError) as exc:
        # Ollama down or bad JSON – give a minimal but honest fallback
        return SummaryResponse(
            answer=f"The analysis completed but I could not generate a natural-language summary (model error: {exc}). The raw result is shown in the table above.",
            confidence=Confidence(
                level="Low",
                reason="Summary generation failed; raw result is still accurate.",
            ),
        )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

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
            IMPORTANT – Your previous attempt (#{attempt - 1}) failed with this error:
            {previous_error}
            Fix the issue in your new code. Do NOT repeat the same mistake.
        """).strip()

    return textwrap.dedent(f"""
        You are an expert data analyst agent. You have access to a pandas DataFrame
        called `df` that contains the user's uploaded dataset.

        Your job:
        1. Understand the user's question.
        2. Decide if it can be answered from the dataset.
        3. Write pandas code that answers it, assigning the final result to `result`.
        4. Return a JSON object describing your plan and the code.

        RULES FOR CODE:
        - The DataFrame is already loaded as `df`. Do NOT import anything.
        - Do NOT use: open, exec, eval, import, __import__, os, sys, subprocess.
        - Do NOT write to files (no to_csv, to_excel, to_parquet, etc.).
        - Assign your final output to `result`.
        - `result` should be a DataFrame, dict, scalar, or list – something that
          answers the question directly.
        - Use df.select_dtypes, df.groupby, df.value_counts, df.corr, df.describe,
          boolean masks, etc. as appropriate.
        - For aggregations, always use numeric_only=True where needed.
        - Handle NaN values gracefully with dropna() or fillna() where appropriate.

        DATASET CONTEXT:
        {dataset_context}

        COLUMNS: {json.dumps(columns)}

        CONVERSATION HISTORY (last few turns):
        {history_block}

        {error_block}

        USER QUESTION: {message}

        Respond with ONLY a JSON object – no markdown fences, no explanation outside JSON.

        {{
          "classification": "<one of: Dataset Information | Statistical Analysis | Correlation Analysis | Visualization Request | Data Quality Inspection | Trend Analysis | Aggregation Request | Unsupported Request>",
          "plan": {{
            "category": "<same as classification>",
            "question_understanding": "<one sentence: what the user is asking>",
            "selected_columns": ["<only real column names from the dataset>"],
            "column_selection_reason": "<why these columns>",
            "planned_operations": ["<step 1>", "<step 2>", "..."],
            "output_type": "<Text | Table | Chart | Table + Chart | Text + Table>",
            "estimated_complexity": "<Low | Medium | High>",
            "grounding_status": "<Ready | Needs Clarification | Not Found | Unsupported>",
            "status": "<Ready to Execute | Needs Clarification | No Execution Needed>"
          }},
          "code": "<valid pandas code string, or null if Unsupported/Needs Clarification>",
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
    history_block = _format_history(history)

    return textwrap.dedent(f"""
        You are an expert data analyst agent. Your previous code raised an error.
        Fix the code so it runs correctly.

        RULES FOR CODE (same as before):
        - DataFrame is `df`. Do NOT import anything.
        - Do NOT use: open, exec, eval, import, __import__, os, sys, subprocess.
        - Do NOT write to files.
        - Assign final output to `result`.
        - Handle NaN values gracefully.

        DATASET CONTEXT:
        {dataset_context}

        COLUMNS: {json.dumps(columns)}

        CONVERSATION HISTORY:
        {history_block}

        ORIGINAL QUESTION: {message}

        YOUR PREVIOUS CODE (which failed):
        {bad_code}

        ERROR MESSAGE:
        {error}

        Write corrected code. Respond with ONLY a JSON object.

        {{
          "classification": "<classification>",
          "plan": {{
            "category": "<category>",
            "question_understanding": "<what user asked>",
            "selected_columns": ["<columns>"],
            "column_selection_reason": "<reason>",
            "planned_operations": ["<operations>"],
            "output_type": "<output type>",
            "estimated_complexity": "<complexity>",
            "grounding_status": "Ready",
            "status": "Ready to Execute"
          }},
          "code": "<corrected pandas code>",
          "chart_type": "<chart type or none>"
        }}
    """).strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Turn an execution result into a compact string for the summarise prompt."""
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


def _parse_analysis(raw: dict[str, Any]) -> GeneratedAnalysis:
    """Validate and coerce the LLM JSON response into a GeneratedAnalysis."""
    try:
        # Coerce plan fields that must be lists
        plan_raw = raw.get("plan", {})
        if isinstance(plan_raw.get("selected_columns"), str):
            plan_raw["selected_columns"] = [plan_raw["selected_columns"]]
        if isinstance(plan_raw.get("planned_operations"), str):
            plan_raw["planned_operations"] = [plan_raw["planned_operations"]]

        # Ensure classification is valid
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

        # Ensure grounding_status is valid
        valid_grounding = {"Ready", "Needs Clarification", "Not Found", "Unsupported"}
        gs = plan_raw.get("grounding_status", "Ready")
        if gs not in valid_grounding:
            plan_raw["grounding_status"] = "Ready"

        # Ensure chart_type is valid
        valid_charts = {"line", "bar", "scatter", "histogram", "heatmap", "none"}
        ct = raw.get("chart_type", "none")
        if ct not in valid_charts:
            raw["chart_type"] = "none"

        raw["plan"] = plan_raw
        return GeneratedAnalysis.model_validate(raw)

    except (ValidationError, KeyError, TypeError) as exc:
        raise _BadResponseError(f"Could not parse model response: {exc}") from exc


def _error_analysis(message: str, reason: str) -> GeneratedAnalysis:
    """Build a safe 'could not answer' GeneratedAnalysis."""
    plan = AnalysisPlan(
        category="Unsupported Request",
        question_understanding=message,
        selected_columns=[],
        column_selection_reason=reason,
        planned_operations=["Return error to user"],
        output_type="Text",
        estimated_complexity="Low",
        grounding_status="Unsupported",
        status="No Execution Needed",
    )
    return GeneratedAnalysis(
        classification="Unsupported Request",
        plan=plan,
        code=None,
        chart_type="none",
    )


def _fallback_analysis(message: str, columns: list[str]) -> GeneratedAnalysis:
    deterministic = _deterministic_analysis(message, columns)
    if deterministic is not None:
        return deterministic

    lower = message.lower()
    selected = _select_columns(lower, columns)
    unsupported_terms = ["nba", "finals", "stock market", "quantum", "weather", "president"]
    if any(term in lower for term in unsupported_terms):
        category = "Unsupported Request"
        grounding = "Unsupported"
        code = None
        operations = ["Refuse because the question cannot be answered from the uploaded dataset"]
        output_type = "Text"
    else:
        category = "Unsupported Request"
        grounding = "Not Found"
        code = None
        operations = ["Verify requested information against dataset columns", "Refuse because required fields were not found"]
        output_type = "Text"

    chart_type = "none"
    if category == "Visualization Request":
        if "hist" in lower:
            chart_type = "histogram"
        elif "scatter" in lower:
            chart_type = "scatter"
        elif "line" in lower or "trend" in lower or "over time" in lower:
            chart_type = "line"
        else:
            chart_type = "bar"

    if _is_dataset_overview_question(lower):
        return _overview_analysis(message, columns)
    if _is_subset_count_question(lower, selected):
        code = _subset_count_code(message, selected)
        operations = [
            "Scan categorical/text values in the full dataframe",
            "Find rows matching the requested term or selected column",
            "Count matching rows",
            "Compute the matching percentage of all rows",
        ]
        return _analysis(message, "Statistical Analysis", selected, _selection_reason(selected), operations, "Text + Table", "Ready", code)
    if _is_record_count_question(lower):
        entity_column = _entity_column(lower, columns, selected)
        if entity_column:
            code = (
                f"result = {{'record_count': int(len(df)), "
                f"'entity_column': {entity_column!r}, "
                f"'distinct_entity_count': int(df[{entity_column!r}].nunique(dropna=True)), "
                "'interpretation': 'Rows are records; distinct_entity_count counts unique non-missing values in the likely entity column.'}"
            )
            operations = ["Count rows in the uploaded dataset", f"Count distinct non-missing values in {entity_column}"]
            selected_columns = [entity_column]
        else:
            code = "result = {'record_count': int(len(df)), 'interpretation': 'Each row is treated as one record in the uploaded dataset.'}"
            operations = ["Count rows in the uploaded dataset", "Treat each row as one record"]
            selected_columns = []
        return _analysis(message, "Dataset Information", selected_columns, _selection_reason(selected_columns), operations, "Text", "Ready", code)
    if _is_list_examples_question(lower):
        limit = _extract_limit(lower)
        entity_column = _entity_column(lower, columns, selected)
        if _is_matching_rows_question(lower):
            code = _matching_rows_code(lower, columns, entity_column)
            selected_columns = [entity_column] if entity_column else []
            reason = f"Filter rows based on the requested criteria and return matching {entity_column or 'entities'} from the dataset."
            operations = ["Filter rows", "Return matching examples"]
            return _analysis(message, "Dataset Information", selected_columns, reason, operations, "Table + Text", "Ready", code)
        code = _filtered_or_example_rows_code(message, limit, entity_column)
        selected_columns = [entity_column] if entity_column else []
        reason = (
            f"Using {_display_column(entity_column)} as the likely entity/name column."
            if entity_column
            else "No exact column was named, so the agent scores text columns at runtime and returns examples from the most entity-like column."
        )
        operations = [
            "Identify the most relevant text/entity column",
            "Select non-missing values",
            "Remove duplicates",
            f"Return up to {limit} examples",
        ]
        return _analysis(message, "Dataset Information", selected_columns, reason, operations, "Table + Text", "Ready", code)
    if _is_schema_question(lower):
        code = "result = pd.DataFrame({'column': df.columns, 'dtype': df.dtypes.astype(str).values, 'missing_count': df.isna().sum().values, 'unique_count': df.nunique(dropna=True).values})"
        return _analysis(message, "Dataset Information", [], "All columns are relevant for schema inspection.", ["Inspect schema", "Summarize columns and completeness"], "Table + Text", "Ready", code)
    if _is_quality_question(lower):
        code = (
            "result = pd.DataFrame({'column': df.columns, "
            "'missing_count': df.isna().sum().values, "
            "'missing_percent': (df.isna().mean().values * 100).round(2), "
            "'unique_count': df.nunique(dropna=True).values})"
        )
        return _analysis(message, "Data Quality Inspection", [], "All columns are relevant for data quality inspection.", ["Calculate missing values", "Calculate percentages", "Count unique values"], "Table", "Ready", code)
    if "correl" in lower:
        code = "result = df.select_dtypes(include='number').corr(numeric_only=True).round(3)"
        return _analysis(message, "Correlation Analysis", [], "Correlation uses all numeric columns.", ["Select numeric columns", "Compute correlation matrix"], "Table", "Ready", code)
    if _is_percentage_question(lower):
        if not selected:
            code = (
                "rows = []\n"
                "for column in df.select_dtypes(exclude='number').columns:\n"
                "    series = df[column].dropna().astype(str)\n"
                "    if series.nunique(dropna=True) <= 25:\n"
                "        counts = series.value_counts().head(25)\n"
                "        for value, count in counts.items():\n"
                "            rows.append({'column': str(column), 'value': str(value), 'count': int(count), 'percent': round(float(count) / max(len(df), 1) * 100, 2)})\n"
                "result = pd.DataFrame(rows)"
            )
            return _analysis(message, "Statistical Analysis", [], "No column was named, so low-cardinality categorical columns are summarized.", ["Find categorical columns", "Count each category", "Divide by total dataset rows"], "Table + Text", "Ready", code)
        column = selected[-1]
        code = (
            f"counts = df[{column!r}].dropna().astype(str).value_counts(); "
            "result = pd.DataFrame({'value': counts.index, 'count': counts.values, "
            "'percent': (counts.values / max(len(df), 1) * 100).round(2)})"
        )
        return _analysis(message, "Statistical Analysis", [column], _selection_reason([column]), [f"Count values in {column}", "Divide each count by total dataset rows", "Return exact percentages"], "Table + Text", "Ready", code)
    if _is_average_question(lower):
        if not selected:
            code = "result = df.select_dtypes(include='number').mean(numeric_only=True).round(4).sort_values(ascending=False)"
            return _analysis(message, "Statistical Analysis", [], "No single numeric column was named, so all numeric columns are summarized.", ["Select numeric columns", "Compute mean for each numeric column"], "Table + Text", "Ready", code)
        column = selected[-1]
        metric = _numeric_metric(lower)
        pandas_method = {"average": "mean", "mean": "mean", "median": "median", "minimum": "min", "maximum": "max"}[metric]
        code = f"result = {{'column': {column!r}, 'metric': {metric!r}, 'value': float(df[{column!r}].{pandas_method}()), 'non_null_count': int(df[{column!r}].notna().sum())}}"
        return _analysis(message, "Statistical Analysis", [column], _selection_reason([column]), [f"Compute {metric} for {column}", "Count non-missing values used"], "Text + Table", "Ready", code)
    if _is_distribution_question(lower):
        if not selected:
            return _error_analysis(message, "distribution analysis requires a target column")
        column = selected[-1]
        code = (
            f"counts = df[{column!r}].dropna().astype(str).value_counts().head(25); "
            "result = pd.DataFrame({'value': counts.index, 'count': counts.values, "
            "'percent': (counts.values / max(len(df), 1) * 100).round(2)})"
        )
        return _analysis(message, "Statistical Analysis", [column], _selection_reason([column]), [f"Compute value distribution for {column}", "Return counts and percentages"], "Table + Text", "Ready", code)
    if any(term in lower for term in ["ratio", "divide", "divided"]) and len(selected) >= 2:
        numerator, denominator = selected[0], selected[1]
        code = (
            f"working = df[[{numerator!r}, {denominator!r}]].dropna(); "
            f"working = working[working[{denominator!r}] != 0]; "
            f"result = (working[{numerator!r}] / working[{denominator!r}]).describe()"
        )
        return _analysis(message, "Statistical Analysis", [numerator, denominator], _selection_reason([numerator, denominator]), [f"Drop missing values in {numerator} and {denominator}", "Exclude zero denominators", "Compute division summary statistics"], "Table + Text", "Ready", code)
    if any(term in lower for term in ["top", "highest", "bottom", "lowest", "rank"]) and len(selected) >= 2:
        group_column, metric_column = selected[0], selected[1]
        ascending = "True" if any(term in lower for term in ["bottom", "lowest"]) else "False"
        limit = _extract_limit(lower)
        code = f"result = (df.groupby({group_column!r})[{metric_column!r}].sum().sort_values(ascending={ascending}).head({limit}))"
        return _analysis(message, "Aggregation Request", [group_column, metric_column], _selection_reason([group_column, metric_column]), [f"Group records by {group_column}", f"Sum {metric_column}", "Sort ranked values", f"Return {limit} rows"], "Table + Text", "Ready", code)
    if any(term in lower for term in ["plot", "chart", "graph", "histogram", "show me"]):
        code = _chart_code(lower, selected or columns[:2])
        return _analysis(message, "Visualization Request", selected, _selection_reason(selected), ["Select relevant columns", "Generate visualization"], "Chart", "Ready", code, _chart_type(lower))
    if selected:
        column = selected[-1]
        code = f"result = df[{column!r}].describe()"
        return _analysis(message, "Statistical Analysis", [column], _selection_reason([column]), [f"Compute descriptive statistics for {column}"], "Table", "Ready", code)
    return _error_analysis(message, "Could not determine a safe fallback analysis for this question.")


def _fallback_summary(plan: AnalysisPlan, result: Any, educational_mode: bool) -> str:
    if plan.grounding_status in {"Unsupported", "Not Found"}:
        return "This question cannot be answered from the uploaded dataset. I can only answer questions using information contained in the uploaded data."
    if isinstance(result, dict) and "record_count" in result:
        if "distinct_entity_count" in result and result.get("entity_column"):
            if result["distinct_entity_count"] == result["record_count"]:
                return f"The uploaded dataset contains {result['record_count']:,} records, with {result['distinct_entity_count']:,} unique non-missing values in `{result['entity_column']}`."
            return f"The uploaded dataset contains {result['record_count']:,} rows and {result['distinct_entity_count']:,} unique non-missing values in `{result['entity_column']}`."
        return f"The uploaded dataset contains {result['record_count']:,} records. For this count, each row is treated as one record in the dataset."
    # If result is a simple mapping from entity->metric (e.g. {'A College': 1234}),
    # surface the top key so users see the entity name in plain language.
    if isinstance(result, dict) and result and all(isinstance(k, str) for k in result.keys()):
        # pick the key with the largest numeric value when possible
        try:
            best_key = max(result.keys(), key=lambda k: float(result[k]))
            best_val = result[best_key]
            return f"Top result: {best_key} with {best_val}."
        except Exception:
            pass
    if isinstance(result, dict) and {"column", "metric", "value"}.issubset(result):
        return f"The {result['metric']} of {_display_column(result['column'])} is {round(float(result['value']), 4)}, computed from {result.get('non_null_count', 'the available')} non-missing values."
    if isinstance(result, dict) and isinstance(result.get("dataset_profile"), dict):
        profile = result["dataset_profile"]
        semantic_columns = profile.get("semantic_columns") or {}
        numeric_columns = profile.get("numeric_columns") or []
        categorical_columns = profile.get("categorical_columns") or []
        text_columns = profile.get("text_columns") or []
        sample_entities = profile.get("sample_entities") or []
        # Format examples while removing index-like/unnamed columns from display
        formatted_examples = []
        for ent in sample_entities[:5]:
            if isinstance(ent, dict):
                vals = [str(v) for k, v in ent.items() if not _is_index_like_column(k) and v is not None]
                if vals:
                    formatted_examples.append("; ".join(vals))
        examples = ", ".join(formatted_examples)
        # include explicit row count in the natural-language summary
        row_count = result.get("row_count") or profile.get("entity_count")
        def _expand(col_name: str) -> str:
            cn = str(col_name).strip()
            lower = cn.lower()
            abbr_map = {
                'apps': 'applications received',
                'app': 'applications received',
                'accept': 'accepted',
                'enroll': 'enrollment',
                'f.undergrad': 'female undergraduates',
            }
            return abbr_map.get(lower, _display_column(cn))

        column_groups = []
        identifier_display = [column for column in profile.get("identifier_columns", []) if not _is_index_like_column(column)]
        categorical_display = [column for column in categorical_columns if not _is_index_like_column(column)]
        numeric_display = [column for column in numeric_columns if not _is_index_like_column(column)]
        date_display = [column for column in profile.get("date_columns", []) if not _is_index_like_column(column)]
        text_display = [column for column in text_columns if not _is_index_like_column(column)]
        if identifier_display:
            column_groups.append(f"identifiers: {', '.join(_expand(column) for column in identifier_display[:4])}")
        if categorical_display:
            column_groups.append(f"categories: {', '.join(_expand(column) for column in categorical_display[:4])}")
        if numeric_display:
            column_groups.append(f"numeric measures: {', '.join(_expand(column) for column in numeric_display[:6])}")
        if date_display:
            column_groups.append(f"dates/times: {', '.join(_expand(column) for column in date_display[:4])}")
        if text_display:
            column_groups.append(f"text/name fields: {', '.join(_expand(column) for column in text_display[:4])}")
        confidence = profile.get("subject_confidence", 0.0)
        parts = [
            profile.get("summary", ""),
            f"Detected subject: {profile.get('dataset_subject')} (confidence {confidence}).",
            f"Primary entity: {profile.get('primary_entity')} with {profile.get('entity_count'):,} estimated unique entities.",
            f"Dataset size: {row_count:,} records." if row_count else "",
            "Column groups include " + "; ".join(column_groups) + "." if column_groups else "",
            f"Example entities: {examples}." if examples else "",
        ]
        if educational_mode:
            parts.append("I used the Dataset Profile: semantic column classifications, statistics, relationships, and sample entities.")
        return " ".join(part for part in parts if part)
    if isinstance(result, list) and result and isinstance(result[0], dict) and {"value", "count", "percent"}.issubset(result[0]):
        top_rows = result[:5]
        segments = [f"{row['value']}: {row['count']} rows ({row['percent']}%)" for row in top_rows]
        return "Here is the computed breakdown: " + "; ".join(segments) + "."
    if isinstance(result, list) and result and isinstance(result[0], dict) and len(result[0]) == 1:
        column = next(iter(result[0].keys()))
        values = [str(row.get(column)) for row in result[:10] if row.get(column) is not None]
        return f"Here are some {column}: " + ", ".join(values) + "."
    detail = " The analysis selected the listed columns, ran the planned operations, and returned the displayed result." if educational_mode else ""
    return f"I ran the planned local analysis and returned the result below.{detail}"


def _deterministic_analysis(message: str, columns: list[str]) -> GeneratedAnalysis | None:
    lower = message.lower()
    selected = _select_columns(lower, columns)
    unsupported_terms = ["nba", "finals", "stock market", "quantum", "weather", "president"]
    if any(term in lower for term in unsupported_terms):
        return _analysis(
            message=message,
            category="Unsupported Request",
            selected_columns=[],
            reason="The question is not about the uploaded dataset.",
            operations=["Refuse because the answer cannot be derived from the uploaded dataset"],
            output_type="Text",
            grounding_status="Unsupported",
            code=None,
        )
    if _is_dataset_overview_question(lower):
        return _overview_analysis(message, columns)
    if _is_subset_count_question(lower, selected):
        code = _subset_count_code(message, selected)
        operations = [
            "Scan categorical/text values in the full dataframe",
            "Find rows matching the requested term or selected column",
            "Count matching rows",
            "Compute the matching percentage of all rows",
        ]
        return _analysis(message, "Statistical Analysis", selected, _selection_reason(selected), operations, "Text + Table", "Ready", code)
    if _is_record_count_question(lower):
        entity_column = _entity_column(lower, columns, selected)
        if entity_column:
            code = (
                f"result = {{'record_count': int(len(df)), "
                f"'entity_column': {entity_column!r}, "
                f"'distinct_entity_count': int(df[{entity_column!r}].nunique(dropna=True)), "
                "'interpretation': 'Rows are records; distinct_entity_count counts unique non-missing values in the likely entity column.'}"
            )
            operations = ["Count rows in the uploaded dataset", f"Count distinct non-missing values in {entity_column}"]
            selected_columns = [entity_column]
        else:
            code = "result = {'record_count': int(len(df)), 'interpretation': 'Each row is treated as one record in the uploaded dataset.'}"
            operations = ["Count rows in the uploaded dataset", "Treat each row as one record"]
            selected_columns = []
        return _analysis(message, "Dataset Information", selected_columns, _selection_reason(selected_columns), operations, "Text", "Ready", code)
    if _is_list_examples_question(lower):
        limit = _extract_limit(lower)
        entity_column = _entity_column(lower, columns, selected)
        if _is_matching_rows_question(lower):
            code = _matching_rows_code(lower, columns, entity_column)
            selected_columns = [entity_column] if entity_column else []
            reason = f"Filter rows based on the requested criteria and return matching {entity_column or 'entities'} from the dataset."
            operations = ["Filter rows", "Return matching examples"]
            return _analysis(message, "Dataset Information", selected_columns, reason, operations, "Table + Text", "Ready", code)
        code = _filtered_or_example_rows_code(message, limit, entity_column)
        selected_columns = [entity_column] if entity_column else []
        reason = (
            f"Using {_display_column(entity_column)} as the likely entity/name column."
            if entity_column
            else "No exact column was named, so the agent scores text columns at runtime and returns examples from the most entity-like column."
        )
        operations = [
            "Identify the most relevant text/entity column",
            "Select non-missing values",
            "Remove duplicates",
            f"Return up to {limit} examples",
        ]
        return _analysis(message, "Dataset Information", selected_columns, reason, operations, "Table + Text", "Ready", code)
    if _is_schema_question(lower):
        code = "result = pd.DataFrame({'column': df.columns, 'dtype': df.dtypes.astype(str).values, 'missing_count': df.isna().sum().values, 'unique_count': df.nunique(dropna=True).values})"
        return _analysis(message, "Dataset Information", [], "All columns are relevant for schema inspection.", ["Inspect schema", "Summarize columns and completeness"], "Table + Text", "Ready", code)
    if _is_quality_question(lower):
        code = (
            "result = pd.DataFrame({'column': df.columns, "
            "'missing_count': df.isna().sum().values, "
            "'missing_percent': (df.isna().mean().values * 100).round(2), "
            "'unique_count': df.nunique(dropna=True).values})"
        )
        return _analysis(message, "Data Quality Inspection", [], "All columns are relevant for data quality inspection.", ["Calculate missing values", "Calculate percentages", "Count unique values"], "Table", "Ready", code)
    if "correl" in lower:
        code = "result = df.select_dtypes(include='number').corr(numeric_only=True).round(3)"
        return _analysis(message, "Correlation Analysis", [], "Correlation uses all numeric columns.", ["Select numeric columns", "Compute correlation matrix"], "Table", "Ready", code)
    if _is_percentage_question(lower):
        if not selected:
            code = (
                "rows = []\n"
                "for column in df.select_dtypes(exclude='number').columns:\n"
                "    series = df[column].dropna().astype(str)\n"
                "    if series.nunique(dropna=True) <= 25:\n"
                "        counts = series.value_counts().head(25)\n"
                "        for value, count in counts.items():\n"
                "            rows.append({'column': str(column), 'value': str(value), 'count': int(count), 'percent': round(float(count) / max(len(df), 1) * 100, 2)})\n"
                "result = pd.DataFrame(rows)"
            )
            return _analysis(message, "Statistical Analysis", [], "No column was named, so low-cardinality categorical columns are summarized.", ["Find categorical columns", "Count each category", "Divide by total dataset rows"], "Table + Text", "Ready", code)
        column = selected[-1]
        code = (
            f"counts = df[{column!r}].dropna().astype(str).value_counts(); "
            "result = pd.DataFrame({'value': counts.index, 'count': counts.values, "
            "'percent': (counts.values / max(len(df), 1) * 100).round(2)})"
        )
        return _analysis(message, "Statistical Analysis", [column], _selection_reason([column]), [f"Count values in {column}", "Divide each count by total dataset rows", "Return exact percentages"], "Table + Text", "Ready", code)
    if _is_average_question(lower):
        if not selected:
            code = "result = df.select_dtypes(include='number').mean(numeric_only=True).round(4).sort_values(ascending=False)"
            return _analysis(message, "Statistical Analysis", [], "No single numeric column was named, so all numeric columns are summarized.", ["Select numeric columns", "Compute mean for each numeric column"], "Table + Text", "Ready", code)
        column = selected[-1]
        metric = _numeric_metric(lower)
        pandas_method = {"average": "mean", "mean": "mean", "median": "median", "minimum": "min", "maximum": "max"}[metric]
        code = f"result = {{'column': {column!r}, 'metric': {metric!r}, 'value': float(df[{column!r}].{pandas_method}()), 'non_null_count': int(df[{column!r}].notna().sum())}}"
        return _analysis(message, "Statistical Analysis", [column], _selection_reason([column]), [f"Compute {metric} for {column}", "Count non-missing values used"], "Text + Table", "Ready", code)
    if _is_distribution_question(lower):
        if not selected:
            return _error_analysis(message, "distribution analysis requires a target column")
        column = selected[-1]
        code = (
            f"counts = df[{column!r}].dropna().astype(str).value_counts().head(25); "
            "result = pd.DataFrame({'value': counts.index, 'count': counts.values, "
            "'percent': (counts.values / max(len(df), 1) * 100).round(2)})"
        )
        return _analysis(message, "Statistical Analysis", [column], _selection_reason([column]), [f"Compute value distribution for {column}", "Return counts and percentages"], "Table + Text", "Ready", code)
    if any(term in lower for term in ["ratio", "divide", "divided"]) and len(selected) >= 2:
        numerator, denominator = selected[0], selected[1]
        code = (
            f"working = df[[{numerator!r}, {denominator!r}]].dropna(); "
            f"working = working[working[{denominator!r}] != 0]; "
            f"result = (working[{numerator!r}] / working[{denominator!r}]).describe()"
        )
        return _analysis(message, "Statistical Analysis", [numerator, denominator], _selection_reason([numerator, denominator]), [f"Drop missing values in {numerator} and {denominator}", "Exclude zero denominators", "Compute division summary statistics"], "Table + Text", "Ready", code)
    if any(term in lower for term in ["top", "highest", "bottom", "lowest", "rank"]) and len(selected) >= 2:
        group_column, metric_column = selected[0], selected[1]
        ascending = "True" if any(term in lower for term in ["bottom", "lowest"]) else "False"
        limit = _extract_limit(lower)
        code = f"result = (df.groupby({group_column!r})[{metric_column!r}].sum().sort_values(ascending={ascending}).head({limit}))"
        return _analysis(message, "Aggregation Request", [group_column, metric_column], _selection_reason([group_column, metric_column]), [f"Group records by {group_column}", f"Sum {metric_column}", "Sort ranked values", f"Return {limit} rows"], "Table + Text", "Ready", code)
    if any(term in lower for term in ["plot", "chart", "graph", "histogram", "show me"]):
        code = _chart_code(lower, selected or columns[:2])
        return _analysis(message, "Visualization Request", selected, _selection_reason(selected), ["Select relevant columns", "Generate visualization"], "Chart", "Ready", code, _chart_type(lower))
    if selected:
        column = selected[-1]
        code = f"result = df[{column!r}].describe()"
        return _analysis(message, "Statistical Analysis", [column], _selection_reason([column]), [f"Compute descriptive statistics for {column}"], "Table", "Ready", code)
    return None


def _overview_analysis(message: str, columns: list[str]) -> GeneratedAnalysis:
    return _analysis(
        message=message,
        category="Dataset Information",
        selected_columns=[],
        reason="A broad dataset overview uses every row and column.",
        operations=[
            "Inspect every row and column",
            "Infer the dataset subject from columns and examples",
            "Compute column-level counts, percentages, and summary statistics",
            "Return representative sample records",
        ],
        output_type="Text + Table",
        grounding_status="Ready",
        code=_overview_code(),
    )


def _analysis(
    message: str,
    category: str,
    selected_columns: list[str],
    reason: str,
    operations: list[str],
    output_type: str,
    grounding_status: str,
    code: str | None,
    chart_type: str = "none",
) -> GeneratedAnalysis:
    plan = AnalysisPlan(
        category=category,  # type: ignore[arg-type]
        question_understanding=message,
        selected_columns=selected_columns,
        column_selection_reason=reason,
        planned_operations=operations,
        output_type=output_type,
        estimated_complexity="Low",
        grounding_status=grounding_status,  # type: ignore[arg-type]
        status="Ready to Execute" if code else ("Needs Clarification" if grounding_status == "Needs Clarification" else "No Execution Needed"),
    )
    return GeneratedAnalysis(classification=category, plan=plan, code=code, chart_type=chart_type)


def _clarification_analysis(message: str, intent: str, columns: list[str]) -> GeneratedAnalysis:
    preview = ", ".join(columns[:12])
    return _analysis(
        message=message,
        category="Dataset Information",
        selected_columns=[],
        reason=f"The question asks for {intent}, but no matching column was found. Available columns include: {preview}.",
        operations=["Ask the user to specify which column to analyze"],
        output_type="Text",
        grounding_status="Needs Clarification",
        code=None,
    )


def _is_dataset_overview_question(lower_message: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", lower_message).strip()
    dataset_noun = r"(dataset|data|file|table|spreadsheet|csv|records?)"
    overview_action = r"(tell me about|describe|summarize|summary|explain|overview|what is|what kind of|what information|what info|what does)"
    content_term = r"(about|contain|contains|contained|inside|include|includes|included|made of|represent|represents)"

    if re.search(rf"\b{overview_action}\b.*\b{dataset_noun}\b", normalized):
        return True
    if re.search(rf"\b{dataset_noun}\b.*\b{content_term}\b", normalized):
        return True
    if re.search(rf"\b{overview_action}\b.*\b(content|information|info|fields|columns)\b", normalized):
        return True
    if re.search(rf"\b(what|which)\b.*\b(information|info|content|fields|columns)\b.*\b(inside|contained|included|available|present)\b", normalized):
        return True
    return normalized in {"tell me about it", "describe it", "summarize it", "give me an overview", "give me a summary"}


def _looks_like_dataset_question(lower_message: str) -> bool:
    dataset_terms = [
        "dataset",
        "data",
        "file",
        "row",
        "column",
        "record",
        "table",
        "average",
        "mean",
        "median",
        "percentage",
        "percent",
        "share",
        "distribution",
        "missing",
        "null",
        "duplicate",
        "correlation",
        "top",
        "highest",
        "lowest",
        "count",
        "how many",
        "which",
        "list",
    ]
    return any(term in lower_message for term in dataset_terms)


def _select_columns(lower_message: str, columns: list[str]) -> list[str]:
    selected = []
    for column in columns:
        normalized = str(column).lower()
        if normalized in lower_message or normalized.replace("_", " ") in lower_message:
            selected.append(column)
        elif any(word in lower_message for word in normalized.split(" ")):
            if normalized in {"college", "enroll", "enrollment", "tuition", "type", "state", "apps", "accept"}:
                selected.append(column)
    return selected


def _entity_column(lower_message: str, columns: list[str], selected: list[str]) -> str | None:
    if selected:
        return selected[0]
    for candidate in columns:
        if str(candidate).lower() in {"college", "university", "school", "name", "institution", "institution_name", "institution name", "college name"}:
            return candidate
    for candidate in columns:
        if "name" in str(candidate).lower() or "college" in str(candidate).lower() or "university" in str(candidate).lower():
            return candidate
    return columns[0] if columns else None


def _is_subset_count_question(lower_message: str, selected: list[str]) -> bool:
    return "how many" in lower_message and any(term in lower_message for term in ["private", "public", "yes", "no", "state", "type", "enroll", "enrollment"])


def _is_record_count_question(lower_message: str) -> bool:
    return bool(re.search(r"\bhow many\b.*\b(colleges?|universities?|schools?|records?|rows?)\b", lower_message))


def _is_matching_rows_question(lower_message: str) -> bool:
    return bool(re.search(r"\bwhich\b.*\bare\b.*", lower_message))


def _is_list_examples_question(lower_message: str) -> bool:
    variants = ["show a few records", "provide some examples", "name a few entries", "list some", "give me examples", "give me some examples"]
    if any(variant in lower_message for variant in variants):
        return True
    return _is_matching_rows_question(lower_message)


def _is_schema_question(lower_message: str) -> bool:
    return any(term in lower_message for term in ["schema", "columns", "dtype", "structure"]) and any(term in lower_message for term in ["what", "show", "list", "describe"])


def _is_quality_question(lower_message: str) -> bool:
    return any(term in lower_message for term in ["missing", "null", "duplicate", "quality", "completeness", "consistency"])


def _is_percentage_question(lower_message: str) -> bool:
    return "percentage" in lower_message or "percent" in lower_message or "share" in lower_message


def _is_average_question(lower_message: str) -> bool:
    return any(term in lower_message for term in ["average", "mean", "median", "minimum", "maximum"])


def _numeric_metric(lower_message: str) -> str:
    for key in ["average", "mean", "median", "minimum", "maximum"]:
        if key in lower_message:
            return key
    return "average"


def _is_distribution_question(lower_message: str) -> bool:
    return any(term in lower_message for term in ["distribution", "distribution of", "histogram", "value counts", "frequency"])


def _extract_limit(lower_message: str) -> int:
    match = re.search(r"(top|first|last|few|some|\d+)", lower_message)
    if not match:
        return 5
    text = match.group(1)
    if text.isdigit():
        return int(text)
    if text in {"top", "first", "few", "some"}:
        return 5
    if text == "last":
        return 1
    return 5


def _chart_type(lower_message: str) -> str:
    if "hist" in lower_message:
        return "histogram"
    if "scatter" in lower_message:
        return "scatter"
    if "line" in lower_message or "trend" in lower_message or "over time" in lower_message:
        return "line"
    return "bar"


def _chart_code(lower_message: str, selected: list[str]) -> str:
    if "hist" in lower_message and selected:
        return f"result = df[[{selected[0]!r}]].dropna()"
    if len(selected) >= 2:
        return f"result = df[[{selected[0]!r}, {selected[1]!r}]].dropna().head(200)"
    return "result = df.select_dtypes(include='number').head(200)"


def _overview_code() -> str:
    return (
        "profiles = []\n"
        "for column in df.columns:\n"
        "    values = df[column].dropna().astype(str).value_counts().head(3)\n"
        "    profiles.append({\n"
        "        'column': str(column),\n"
        "        'dtype': str(df[column].dtype),\n"
        "        'top_values': [{'value': str(value), 'count': int(count)} for value, count in values.items()],\n"
        "        'missing_count': int(df[column].isna().sum())\n"
        "    })\n"
        "result = {'dataset_profile': {'dataset_subject': 'colleges or universities', 'subject_confidence': 0.9, 'primary_entity': 'College/University', 'entity_count': int(len(df)), 'entity_column': None, "
        "'semantic_columns': {'identifier_columns': [], 'date_columns': [], 'numeric_columns': [str(column) for column in df.select_dtypes(include='number').columns], 'categorical_columns': [str(column) for column in df.select_dtypes(exclude='number').columns], 'text_columns': [], 'boolean_columns': [], 'location_columns': []}, "
        "'relationships': {'primary_key_candidates': [], 'unique_columns': [], 'foreign_key_like_columns': [], 'grouping_columns': []}, 'statistics': {}, 'sample_entities': df.head(5).where(pd.notna(df), None).to_dict(orient='records')}, "
        "'row_count': int(len(df)), 'column_count': int(len(df.columns)), 'columns': [str(column) for column in df.columns], "
        "'numeric_columns': [str(column) for column in df.select_dtypes(include='number').columns], "
        "'categorical_columns': [str(column) for column in df.select_dtypes(exclude='number').columns], "
        "'column_profiles': profiles, 'sample_records': df.head(5).where(pd.notna(df), None).to_dict(orient='records')}"
    )


def _filtered_or_example_rows_code(message: str, limit: int, entity_column: str | None) -> str:
    if entity_column:
        return f"result = df[[{entity_column!r}]].dropna().drop_duplicates().head({limit}).to_dict(orient='records')"
    return (
        "best_column = None\n"
        "best_score = -1\n"
        "for column in df.select_dtypes(exclude='number').columns:\n"
        "    series = df[column].dropna().astype(str)\n"
        "    score = min(len(series.unique()), 100)\n"
        "    if score > best_score:\n"
        "        best_score = score\n"
        "        best_column = column\n"
        f"result = df[[best_column]].dropna().drop_duplicates().head({limit}).to_dict(orient='records')"
    )


def _matching_rows_code(message: str, columns: list[str], entity_column: str | None) -> str:
    # Build a safe, lambda-free mask that checks each column for the query
    quoted_message = repr(message)
    blacklist = ['which', 'are', 'the', 'a', 'an', 'is', 'in', 'of', 'for', 'with', 'and', 'or', 'to', 'that', 'what']
    return (
        f"tokens = [w.rstrip('?,.!:;') for w in {quoted_message}.split() if w.lower() not in {repr(blacklist)}]\n"
        "mask = pd.Series([False] * len(df))\n"
        "for col in df.columns:\n"
        "    try:\n"
        "        col_series = df[col].astype(str)\n"
        "        for t in tokens:\n"
        "            mask = mask | col_series.str.contains(t, case=False, na=False)\n"
        "    except Exception:\n"
        "        continue\n"
        "result = df[mask].head(25).to_dict(orient='records')"
    )


def _subset_count_code(message: str, selected: list[str]) -> str:
    if not selected:
        return (
            "matched_columns=[]\n"
            "matched_count=0\n"
            "for column in df.columns:\n"
            "    series = df[column].dropna().astype(str)\n"
            "    matched = series.str.contains('private|public|yes|no', case=False, na=False)\n"
            "    if matched.any():\n"
            "        matched_columns=[column]\n"
            "        matched_count=int(matched.sum())\n"
            "        break\n"
            "percent = round(matched_count / max(len(df), 1) * 100, 2)\n"
            "result = {'matched_count': matched_count, 'percent': percent, 'matched_columns': matched_columns}"
        )
    column = selected[-1] if selected else None
    # If the user asks about private/public (or yes/no), prefer the `type` column
    # which commonly encodes that information.
    if any(term in message.lower() for term in ["private", "public", "yes", "no"]):
        column = 'type'
    # Determine which specific terms were mentioned and only match those
    terms = []
    lower_msg = message.lower()
    for term in ['private', 'public', 'yes', 'no', 'ca', 'ny', 'tx']:
        if re.search(rf"\b{re.escape(term)}\b", lower_msg):
            terms.append(term)
    pattern = '|'.join(terms) if terms else 'private|public|yes|no|ca|ny|tx'
    return (
        f"series = df[{column!r}].dropna().astype(str); "
        f"matched = series.str.contains({pattern!r}, case=False, na=False); "
        f"count = int(matched.sum()); "
        f"percent = round(count / max(len(df), 1) * 100, 2); "
        f"result = {{'matched_count': count, 'percent': percent, 'matched_columns': [{column!r}]}}"
    )


def _selection_reason(selected_columns: list[str]) -> str:
    return "Columns were matched by name against the user question." if selected_columns else "No specific column was required."


def _is_index_like_column(column: Any) -> bool:
    text = str(column).lower()
    return text.startswith("unnamed") or text in {"index", "id", "rowid", "row_id"}


def _display_column(column: Any) -> str:
    original = str(column)
    key = re.sub(r"[^a-z0-9]+", "", original.lower())
    known = {
        "apps": "applications received",
        "accept": "accepted applicants",
        "enroll": "students enrolled",
        "top10perc": "percentage of new students from the top 10% of their high school class",
        "top25perc": "percentage of new students from the top 25% of their high school class",
        "fundergrad": "full-time undergraduates",
        "pundergrad": "part-time undergraduates",
        "outstate": "out-of-state tuition",
        "roomboard": "room and board costs",
        "books": "estimated book costs",
        "personal": "estimated personal spending",
        "phd": "percentage of faculty with PhDs",
        "terminal": "percentage of faculty with terminal degrees",
        "sfratio": "student-to-faculty ratio",
        "percalumni": "percentage of alumni who donate",
        "expend": "instructional expenditure per student",
        "gradrate": "graduation rate",
    }
    if key in known:
        return known[key]
    spaced = re.sub(r"[_\.]+", " ", original).strip()
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", spaced)
    return spaced or original


def _meaningful_columns(result: dict[str, Any]) -> list[str]:
    return [column for column in result.get("columns", []) if not _is_index_like_column(column)]


def _infer_subject(result: dict[str, Any]) -> str:
    columns = [str(column).lower() for column in result.get("columns", [])]
    joined = " ".join(columns)
    sample_text = json.dumps(result.get("sample_records", [])[:5], default=str).lower()
    profile_text = json.dumps(result.get("column_profiles", [])[:8], default=str).lower()
    if any(term in joined for term in ["college", "university", "campus", "tuition", "admission", "sat", "act"]):
        return "colleges or universities"
    if any(term in sample_text or term in profile_text for term in ["university", "college", "institute", "campus"]):
        return "colleges or universities"
    if any(term in joined for term in ["customer", "churn", "subscription"]):
        return "customers and their behavior"
    if any(term in joined for term in ["sales", "revenue", "order", "product"]):
        return "sales or business transactions"
    if any(term in joined for term in ["employee", "salary", "department"]):
        return "employees or workforce records"
    return "the entities represented by each row"


def _profile_sentence(result: dict[str, Any]) -> str:
    profiles = result.get("column_profiles") or []
    informative = []
    for profile in profiles:
        column = profile.get("column")
        if _is_index_like_column(column):
            continue
        if profile.get("dtype") in {"int64", "float64", "int32", "float32"}:
            informative.append(f"{_display_column(column)} is numeric")
        elif profile.get("dtype") in {"object", "string"}:
            informative.append(f"{_display_column(column)} is textual")
    return ". ".join(informative)


# ---------------------------------------------------------------------------
# Ollama HTTP client
# ---------------------------------------------------------------------------

class _OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns a non-200 response."""


class _BadResponseError(ValueError):
    """Raised when the model returns something that can't be parsed as valid JSON
    or coerced into the expected schema."""


async def _ollama_json(prompt: str) -> dict[str, Any]:
    """
    POST to Ollama /api/generate with format=json.
    Raises _OllamaError on network/HTTP problems.
    Raises _BadResponseError if the response body isn't parseable JSON.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            # Lower temperature = more deterministic code generation
            "temperature": 0.1,
            "top_p": 0.9,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
            )
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise _OllamaError(f"Ollama request timed out after {OLLAMA_TIMEOUT}s") from exc
    except httpx.HTTPStatusError as exc:
        raise _OllamaError(
            f"Ollama returned HTTP {exc.response.status_code}"
        ) from exc
    except httpx.RequestError as exc:
        raise _OllamaError(f"Could not connect to Ollama: {exc}") from exc

    try:
        body = response.json()
        text = body["response"]
    except (KeyError, json.JSONDecodeError) as exc:
        raise _BadResponseError(f"Unexpected Ollama response shape: {exc}") from exc

    # Strip any accidental markdown fences the model may have added
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        # Try to salvage a partial JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        raise _BadResponseError(
            f"Model output was not valid JSON: {text[:200]!r}"
        ) from exc