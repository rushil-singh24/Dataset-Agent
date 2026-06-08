from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
import pandas as pd
from pydantic import ValidationError

from .schemas import AnalysisPlan, Confidence, GeneratedAnalysis, SummaryResponse


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")


async def build_analysis(message: str, dataset_context: str, columns: list[str], history: list[dict[str, Any]]) -> GeneratedAnalysis:
    deterministic = _deterministic_analysis(message, columns)
    if deterministic is not None:
        return deterministic

    prompt = f"""
You are a local data analysis agent. Return strict JSON only.
You must answer only from the uploaded dataset. Never invent columns.

Dataset context:
{dataset_context}

Columns:
{json.dumps(columns)}

Recent conversation:
{json.dumps(history[-6:])}

User question:
{message}

Return this JSON shape:
{{
  "classification": "Dataset Information|Statistical Analysis|Correlation Analysis|Visualization Request|Data Quality Inspection|Trend Analysis|Aggregation Request|Unsupported Request",
  "plan": {{
    "category": "...same classification...",
    "question_understanding": "short text",
    "selected_columns": ["real column names only"],
    "column_selection_reason": "short text",
    "planned_operations": ["operation"],
    "output_type": "Text|Table|Chart|Table + Chart",
    "estimated_complexity": "Low|Medium|High",
    "grounding_status": "Ready|Needs Clarification|Not Found|Unsupported",
    "status": "Ready to Execute"
  }},
  "code": "Pandas code that assigns final output to result, or null if unsupported/not found. The dataframe df contains the full uploaded dataset: all rows and all columns. Do not import anything.",
  "chart_type": "line|bar|scatter|histogram|heatmap|none"
}}
"""
    try:
        data = await _ollama_json(prompt)
        analysis = GeneratedAnalysis.model_validate(data)
        lower = message.lower()
        if (
            analysis.classification == "Unsupported Request"
            or analysis.plan.grounding_status in {"Unsupported", "Not Found"}
        ) and _looks_like_dataset_question(lower):
            return _overview_analysis(message, columns)
        if analysis.classification == "Unsupported Request":
            analysis.code = None
        return analysis
    except (httpx.HTTPError, ValidationError, json.JSONDecodeError, KeyError):
        return _fallback_analysis(message, columns)


async def summarize_result(
    message: str,
    plan: AnalysisPlan,
    result: Any,
    educational_mode: bool,
    missing_relevant_values: bool,
) -> SummaryResponse:
    prompt = f"""
Return strict JSON only. Explain the answer using only the execution result.
User question: {message}
Analysis plan: {plan.model_dump_json()}
Execution result: {json.dumps(result, default=str)[:12000]}
Educational mode: {educational_mode}

Return:
{{
  "answer": "grounded answer",
  "confidence": {{"level": "High|Medium|Low", "reason": "short reason"}}
}}
"""
    try:
        data = await _ollama_json(prompt)
        return SummaryResponse.model_validate(data)
    except (httpx.HTTPError, ValidationError, json.JSONDecodeError, KeyError):
        confidence = Confidence(
            level="Medium" if missing_relevant_values else "High",
            reason="Answer is based on executed local analysis; confidence is reduced when relevant columns contain missing values."
            if missing_relevant_values
            else "Question was answered from existing dataset columns and executed local analysis.",
        )
        return SummaryResponse(answer=_fallback_summary(plan, result, educational_mode), confidence=confidence)


async def _ollama_json(prompt: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"},
        )
        response.raise_for_status()
    payload = response.json()
    return json.loads(payload["response"])


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

    plan = AnalysisPlan(
        category=category,
        question_understanding=message,
        selected_columns=selected,
        column_selection_reason="Columns were matched by name against the user question." if selected else "No specific columns were required or confidently matched.",
        planned_operations=operations,
        output_type=output_type,
        estimated_complexity="Low",
        grounding_status=grounding,
        status="Ready to Execute" if code else "No Execution Needed",
    )
    return GeneratedAnalysis(classification=category, plan=plan, code=code, chart_type=chart_type)


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
            return _clarification_analysis(message, "distribution analysis", columns)
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
    return GeneratedAnalysis(classification=category, plan=plan, code=code, chart_type=chart_type)  # type: ignore[arg-type]


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
    if re.search(r"\b(what|which)\b.*\b(information|info|content|fields|columns)\b.*\b(inside|contained|included|available|present)\b", normalized):
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
        "show",
        "tell me",
        "describe",
        "summarize",
    ]
    return any(term in lower_message for term in dataset_terms)


def _is_record_count_question(lower_message: str) -> bool:
    if any(term in lower_message for term in ["missing", "null", "duplicate", "quality"]):
        return False
    if any(term in lower_message for term in ["how many rows", "number of rows", "row count", "count rows"]):
        return True
    return bool(re.search(r"\b(how many|number of|count of)\b", lower_message))


def _is_subset_count_question(lower_message: str, selected: list[str]) -> bool:
    if not re.search(r"\b(how many|number of|count of)\b", lower_message):
        return False
    if any(term in lower_message for term in ["rows", "records", "columns", "variables", "fields"]):
        return False
    filter_tokens = _meaningful_query_tokens(lower_message)
    if not filter_tokens:
        return False
    if selected:
        return True
    return bool(filter_tokens)


def _is_list_examples_question(lower_message: str) -> bool:
    has_listing_action = bool(re.search(r"\b(list|show|give|name|provide|sample|examples?|which)\b", lower_message))
    asks_for_subset = bool(re.search(r"\b(some|few|several|examples?|sample|names?)\b", lower_message))
    dataset_scoped = any(term in lower_message for term in ["dataset", "data", "file", "rows", "records"])
    asks_for_entities = bool(re.search(r"\b(colleges?|universities|schools?|customers?|products?|employees?|students?|companies?|entries|items)\b", lower_message))
    return has_listing_action and (asks_for_subset or dataset_scoped or asks_for_entities)


def _is_schema_question(lower_message: str) -> bool:
    return any(term in lower_message for term in ["columns", "fields", "schema", "dtypes", "data types", "what variables"])


def _is_quality_question(lower_message: str) -> bool:
    return any(term in lower_message for term in ["missing", "null", "duplicate", "quality", "empty column", "constant column"])


def _is_percentage_question(lower_message: str) -> bool:
    return any(term in lower_message for term in ["percentage", "percent", "proportion", "share"])


def _is_average_question(lower_message: str) -> bool:
    return any(term in lower_message for term in ["average", "mean", "median", "minimum", "maximum", "min ", "max "])


def _is_distribution_question(lower_message: str) -> bool:
    return any(term in lower_message for term in ["distribution", "breakdown", "split by", "by category", "value counts", "frequency"])


def _select_columns(lower_message: str, columns: list[str]) -> list[str]:
    selected = []
    normalized = _normalize_text(lower_message)
    message_tokens = set(normalized.split())
    for column in columns:
        token = _normalize_text(column)
        display_token = _normalize_text(_display_column(column))
        column_tokens = set(token.split()) | set(display_token.split())
        if token and token in normalized:
            selected.append(column)
        elif display_token and display_token in normalized:
            selected.append(column)
        elif column_tokens and column_tokens.difference(_COLUMN_STOPWORDS).intersection(message_tokens):
            selected.append(column)
    return selected[:8]


def _selection_reason(selected_columns: list[str]) -> str:
    return "Columns were matched by name against the user question." if selected_columns else "No specific column was required."


_COLUMN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "column",
    "columns",
    "data",
    "dataset",
    "field",
    "fields",
    "file",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "record",
    "records",
    "row",
    "rows",
    "table",
    "the",
    "this",
    "to",
    "with",
}

_QUERY_STOPWORDS = _COLUMN_STOPWORDS | {
    "about",
    "count",
    "does",
    "entries",
    "entry",
    "examples",
    "few",
    "give",
    "how",
    "include",
    "inside",
    "items",
    "list",
    "many",
    "me",
    "name",
    "number",
    "provide",
    "sample",
    "several",
    "show",
    "some",
    "tell",
    "there",
    "unique",
    "what",
    "which",
}

_ENTITY_WORDS = {
    "college",
    "colleges",
    "universities",
    "university",
    "school",
    "schools",
    "customer",
    "customers",
    "employee",
    "employees",
    "student",
    "students",
    "product",
    "products",
    "company",
    "companies",
    "entity",
    "entities",
}


def _normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _meaningful_query_tokens(message: str) -> list[str]:
    normalized = _normalize_text(message)
    tokens = []
    for token in normalized.split():
        if token in _QUERY_STOPWORDS or token in _ENTITY_WORDS:
            continue
        if len(token) < 2:
            continue
        tokens.append(token)
    return tokens[:8]


def _entity_column(lower_message: str, columns: list[str], selected: list[str]) -> str | None:
    if "unique" in lower_message and selected:
        return selected[-1]
    entity_terms = ["college", "university", "school", "customer", "employee", "student", "product", "company", "name"]
    for column in selected:
        normalized = re.sub(r"[^a-z0-9]+", " ", column.lower())
        if any(term in normalized for term in entity_terms):
            return column
    for column in columns:
        normalized = re.sub(r"[^a-z0-9]+", " ", column.lower())
        if any(term in normalized for term in entity_terms):
            return column
    for column in columns:
        if _is_index_like_column(column):
            return column
    return selected[-1] if selected else None


def _numeric_metric(lower_message: str) -> str:
    if "median" in lower_message:
        return "median"
    if "minimum" in lower_message or re.search(r"\bmin\b", lower_message):
        return "minimum"
    if "maximum" in lower_message or re.search(r"\bmax\b", lower_message):
        return "maximum"
    if "mean" in lower_message:
        return "mean"
    return "average"


def _extract_limit(lower_message: str) -> int:
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    number_match = re.search(r"\b(\d{1,2})\b", lower_message)
    if number_match:
        return min(max(int(number_match.group(1)), 1), 25)
    for word, value in words.items():
        if word in lower_message:
            return value
    return 5


def _list_examples_code(limit: int, preferred_column: str | None) -> str:
    if preferred_column:
        display = _display_column(preferred_column)
        return f"result = pd.DataFrame({{{display!r}: df[{preferred_column!r}].dropna().astype(str).drop_duplicates().head({limit}).values}})"

    return (
        "best_column = None\n"
        "best_score = -1\n"
        "for column in df.columns:\n"
        "    series = df[column].dropna().astype(str)\n"
        "    if len(series) == 0:\n"
        "        continue\n"
        "    unique_count = int(series.nunique(dropna=True))\n"
        "    unique_ratio = unique_count / max(len(df), 1)\n"
        "    avg_len = float(series.str.len().mean())\n"
        "    low_cardinality_penalty = 25 if unique_count <= 3 else 0\n"
        "    score = unique_ratio * 100 + min(avg_len, 40) - low_cardinality_penalty\n"
        "    if score > best_score:\n"
        "        best_score = score\n"
        "        best_column = column\n"
        "if best_column is None:\n"
        f"    result = df.head({limit})\n"
        "else:\n"
        f"    result = pd.DataFrame({{'examples': df[best_column].dropna().astype(str).drop_duplicates().head({limit}).values}})\n"
    )


def _subset_count_code(message: str, selected: list[str]) -> str:
    tokens = _meaningful_query_tokens(message)
    target_columns = [column for column in selected if not _is_entity_named_column(column)]
    return (
        f"query_tokens = {tokens!r}\n"
        f"preferred_columns = {target_columns!r}\n"
        "candidate_columns = preferred_columns if preferred_columns else [column for column in df.columns if df[column].dtype == 'object' or str(df[column].dtype).startswith('category')]\n"
        "mask = pd.Series(False, index=df.index)\n"
        "matched_columns = []\n"
        "for column in candidate_columns:\n"
        "    if column not in df.columns:\n"
        "        continue\n"
        "    text_values = df[column].astype(str).str.lower()\n"
        "    column_text = str(column).lower()\n"
        "    column_matched = False\n"
        "    yes_like = text_values.isin(['yes', 'true', '1', 'y'])\n"
        "    if bool(yes_like.any()):\n"
        "        for token in query_tokens:\n"
        "            if token in column_text:\n"
        "                mask = mask | yes_like\n"
        "                column_matched = True\n"
        "    for token in query_tokens:\n"
        "        contains_token = text_values.str.contains(token, regex=False, na=False)\n"
        "        if bool(contains_token.any()):\n"
        "            mask = mask | contains_token\n"
        "            column_matched = True\n"
        "    if column_matched:\n"
        "        matched_columns.append(str(column))\n"
        "matched_count = int(mask.sum())\n"
        "result = {\n"
        "    'matched_count': matched_count,\n"
        "    'total_rows': int(len(df)),\n"
        "    'percent': round(float(matched_count) / max(len(df), 1) * 100, 2),\n"
        "    'matched_columns': matched_columns,\n"
        "    'query_terms': query_tokens,\n"
        "}\n"
    )


def _filtered_or_example_rows_code(message: str, limit: int, preferred_column: str | None) -> str:
    tokens = _meaningful_query_tokens(message)
    if preferred_column and not tokens:
        return _list_examples_code(limit, preferred_column)
    preferred_columns = []
    return (
        f"query_tokens = {tokens!r}\n"
        f"preferred_columns = {preferred_columns!r}\n"
        "candidate_columns = preferred_columns if preferred_columns else [column for column in df.columns if df[column].dtype == 'object' or str(df[column].dtype).startswith('category')]\n"
        "mask = pd.Series(False, index=df.index)\n"
        "matched_columns = []\n"
        "for column in candidate_columns:\n"
        "    if column not in df.columns:\n"
        "        continue\n"
        "    text_values = df[column].astype(str).str.lower()\n"
        "    column_text = str(column).lower()\n"
        "    column_matched = False\n"
        "    yes_like = text_values.isin(['yes', 'true', '1', 'y'])\n"
        "    if bool(yes_like.any()):\n"
        "        for token in query_tokens:\n"
        "            if token in column_text:\n"
        "                mask = mask | yes_like\n"
        "                column_matched = True\n"
        "    for token in query_tokens:\n"
        "        contains_token = text_values.str.contains(token, regex=False, na=False)\n"
        "        if bool(contains_token.any()):\n"
        "            mask = mask | contains_token\n"
        "            column_matched = True\n"
        "    if column_matched:\n"
        "        matched_columns.append(str(column))\n"
        "if query_tokens and bool(mask.any()):\n"
        f"    result = df.loc[mask].head({limit})\n"
        "else:\n"
        "    best_column = None\n"
        "    best_score = -1\n"
        "    for column in df.columns:\n"
        "        series = df[column].dropna().astype(str)\n"
        "        if len(series) == 0:\n"
        "            continue\n"
        "        unique_count = int(series.nunique(dropna=True))\n"
        "        unique_ratio = unique_count / max(len(df), 1)\n"
        "        avg_len = float(series.str.len().mean())\n"
        "        low_cardinality_penalty = 25 if unique_count <= 3 else 0\n"
        "        score = unique_ratio * 100 + min(avg_len, 40) - low_cardinality_penalty\n"
        "        if score > best_score:\n"
        "            best_score = score\n"
        "            best_column = column\n"
        "    if best_column is None:\n"
        f"        result = df.head({limit})\n"
        "    else:\n"
        f"        result = pd.DataFrame({{'examples': df[best_column].dropna().astype(str).drop_duplicates().head({limit}).values}})\n"
    )


def _chart_type(lower_message: str) -> str:
    if "hist" in lower_message:
        return "histogram"
    if "scatter" in lower_message:
        return "scatter"
    if "heatmap" in lower_message:
        return "heatmap"
    if "line" in lower_message or "trend" in lower_message or "over time" in lower_message:
        return "line"
    return "bar"


def _overview_code() -> str:
    return (
        "row_count = int(len(df))\n"
        "column_profiles = []\n"
        "semantic_columns = {\n"
        "    'identifier_columns': [],\n"
        "    'name_columns': [],\n"
        "    'person_columns': [],\n"
        "    'organization_columns': [],\n"
        "    'location_columns': [],\n"
        "    'date_columns': [],\n"
        "    'numeric_columns': [],\n"
        "    'categorical_columns': [],\n"
        "    'text_columns': [],\n"
        "    'boolean_columns': [],\n"
        "    'unknown_columns': [],\n"
        "}\n"
        "statistics = {}\n"
        "unique_columns = []\n"
        "primary_key_candidates = []\n"
        "grouping_columns = []\n"
        "for column in df.columns:\n"
        "    series = df[column]\n"
        "    column_name = str(column)\n"
        "    normalized_name = column_name.lower().replace('_', ' ').replace('.', ' ')\n"
        "    non_null = int(series.notna().sum())\n"
        "    unique_count = int(series.nunique(dropna=True))\n"
        "    unique_ratio = float(unique_count) / max(row_count, 1)\n"
        "    sample_values = [str(value) for value in series.dropna().head(5).tolist()]\n"
        "    lowered_samples = ' '.join(sample_values).lower()\n"
        "    semantic_type = 'Unknown'\n"
        "    is_bool_like = unique_count <= 2 and series.dropna().astype(str).str.lower().isin(['yes', 'no', 'true', 'false', '0', '1', 'y', 'n']).all()\n"
        "    if 'id' == normalized_name.strip() or normalized_name.endswith(' id') or normalized_name.endswith('id') or ' identifier' in normalized_name:\n"
        "        semantic_type = 'Identifier'\n"
        "    elif any(term in normalized_name for term in ['first name', 'last name', 'full name', 'name', 'title']) and not any(term in normalized_name for term in ['company', 'organization', 'institution']):\n"
        "        semantic_type = 'Name'\n"
        "    elif any(term in normalized_name for term in ['person', 'patient', 'employee', 'student', 'author']):\n"
        "        semantic_type = 'Person'\n"
        "    elif any(term in normalized_name for term in ['company', 'organization', 'organisation', 'college', 'university', 'school', 'institution', 'vendor', 'employer']):\n"
        "        semantic_type = 'Organization'\n"
        "    elif any(term in normalized_name for term in ['country', 'state', 'city', 'county', 'zip', 'postal', 'address', 'region', 'location', 'latitude', 'longitude']):\n"
        "        semantic_type = 'Location'\n"
        "    elif any(term in normalized_name for term in ['date', 'time', 'year', 'month', 'created', 'updated']) or str(series.dtype).startswith('datetime'):\n"
        "        semantic_type = 'Date/Time'\n"
        "    elif is_bool_like:\n"
        "        semantic_type = 'Boolean'\n"
        "    elif series.dtype.kind in 'biufc':\n"
        "        semantic_type = 'Numeric Measure'\n"
        "    elif unique_count <= min(50, max(10, int(row_count * 0.2))):\n"
        "        semantic_type = 'Category'\n"
        "    elif series.dropna().astype(str).str.len().mean() > 45:\n"
        "        semantic_type = 'Text Description'\n"
        "    elif any(term in lowered_samples for term in ['university', 'college', 'inc', 'llc', 'school', 'hospital']):\n"
        "        semantic_type = 'Organization'\n"
        "    else:\n"
        "        semantic_type = 'Text Description'\n"
        "    if unique_count == row_count and row_count > 0:\n"
        "        unique_columns.append(column_name)\n"
        "        if semantic_type in ['Identifier', 'Name', 'Organization'] or unique_ratio >= 0.98:\n"
        "            primary_key_candidates.append(column_name)\n"
        "    if semantic_type in ['Category', 'Boolean', 'Location'] and 1 < unique_count <= min(50, max(10, int(row_count * 0.5))):\n"
        "        grouping_columns.append(column_name)\n"
        "    semantic_key = {\n"
        "        'Identifier': 'identifier_columns',\n"
        "        'Name': 'name_columns',\n"
        "        'Person': 'person_columns',\n"
        "        'Organization': 'organization_columns',\n"
        "        'Location': 'location_columns',\n"
        "        'Date/Time': 'date_columns',\n"
        "        'Numeric Measure': 'numeric_columns',\n"
        "        'Category': 'categorical_columns',\n"
        "        'Text Description': 'text_columns',\n"
        "        'Boolean': 'boolean_columns',\n"
        "        'Unknown': 'unknown_columns',\n"
        "    }[semantic_type]\n"
        "    semantic_columns[semantic_key].append(column_name)\n"
        "    profile = {\n"
        "        'column': column_name,\n"
        "        'dtype': str(series.dtype),\n"
        "        'semantic_type': semantic_type,\n"
        "        'non_null_count': non_null,\n"
        "        'missing_count': int(series.isna().sum()),\n"
        "        'missing_percent': round(float(series.isna().mean() * 100), 2),\n"
        "        'unique_count': unique_count,\n"
        "        'examples': sample_values[:3],\n"
        "    }\n"
        "    if series.dtype.kind in 'biufc':\n"
        "        profile['mean'] = None if pd.isna(series.mean()) else float(series.mean())\n"
        "        profile['median'] = None if pd.isna(series.median()) else float(series.median())\n"
        "        profile['std'] = None if pd.isna(series.std()) else float(series.std())\n"
        "        profile['min'] = None if pd.isna(series.min()) else float(series.min())\n"
        "        profile['max'] = None if pd.isna(series.max()) else float(series.max())\n"
        "        statistics[column_name] = {'min': profile['min'], 'max': profile['max'], 'mean': profile['mean'], 'median': profile['median'], 'std': profile['std']}\n"
        "    else:\n"
        "        counts = series.dropna().astype(str).value_counts().head(5)\n"
        "        profile['top_values'] = [\n"
        "            {'value': str(index), 'count': int(count), 'percent': round(float(count) / max(len(df), 1) * 100, 2)}\n"
        "            for index, count in counts.items()\n"
        "        ]\n"
        "        statistics[column_name] = {'distinct_values': unique_count, 'top_values': profile['top_values']}\n"
        "        if semantic_type == 'Text Description':\n"
        "            statistics[column_name]['average_length'] = round(float(series.dropna().astype(str).str.len().mean()), 2) if non_null else 0\n"
        "    column_profiles.append(profile)\n"
        "all_text = (' '.join([str(column) for column in df.columns]) + ' ' + str(df.head(5).to_dict(orient='records'))).lower()\n"
        "dataset_subject = 'the entities represented by each row'\n"
        "subject_confidence = 0.45\n"
        "subject_reasoning = 'The subject was inferred from column names, semantic column types, and sample values.'\n"
        "primary_entity = 'Record'\n"
        "if any(term in all_text for term in ['college', 'university', 'campus', 'tuition', 'admission', 'sat', 'act']):\n"
        "    dataset_subject = 'colleges or universities'\n"
        "    primary_entity = 'College/University'\n"
        "    subject_confidence = 0.9\n"
        "elif any(term in all_text for term in ['customer', 'churn', 'subscription', 'account']):\n"
        "    dataset_subject = 'customer records'\n"
        "    primary_entity = 'Customer'\n"
        "    subject_confidence = 0.85\n"
        "elif any(term in all_text for term in ['order', 'invoice', 'transaction', 'payment', 'revenue', 'sales']):\n"
        "    dataset_subject = 'business transactions or sales records'\n"
        "    primary_entity = 'Transaction'\n"
        "    subject_confidence = 0.8\n"
        "elif any(term in all_text for term in ['employee', 'salary', 'department', 'job title']):\n"
        "    dataset_subject = 'employee or workforce records'\n"
        "    primary_entity = 'Employee'\n"
        "    subject_confidence = 0.8\n"
        "elif any(term in all_text for term in ['product', 'sku', 'category', 'price', 'inventory']):\n"
        "    dataset_subject = 'product catalog or inventory records'\n"
        "    primary_entity = 'Product'\n"
        "    subject_confidence = 0.8\n"
        "elif any(term in all_text for term in ['patient', 'diagnosis', 'medical', 'healthcare', 'hospital']):\n"
        "    dataset_subject = 'healthcare or patient records'\n"
        "    primary_entity = 'Patient/Healthcare Record'\n"
        "    subject_confidence = 0.78\n"
        "entity_column = None\n"
        "for column in primary_key_candidates + semantic_columns['name_columns'] + semantic_columns['organization_columns'] + semantic_columns['identifier_columns']:\n"
        "    if column in df.columns:\n"
        "        entity_column = column\n"
        "        break\n"
        "entity_count = int(df[entity_column].nunique(dropna=True)) if entity_column else row_count\n"
        "relationships = {\n"
        "    'primary_key_candidates': primary_key_candidates,\n"
        "    'unique_columns': unique_columns,\n"
        "    'foreign_key_like_columns': [column for column in semantic_columns['identifier_columns'] if column not in primary_key_candidates],\n"
        "    'grouping_columns': grouping_columns,\n"
        "}\n"
        "summary = (\n"
        "    f'This dataset appears to be about {dataset_subject}. '\n"
        "    f'It contains {row_count:,} records and {int(len(df.columns)):,} columns. '\n"
        "    f'Each row most likely represents a {primary_entity.lower()}. '\n"
        "    f'The profile found {len(semantic_columns[\"identifier_columns\"])} identifier fields, '\n"
        "    f'{len(semantic_columns[\"categorical_columns\"])} categorical fields, '\n"
        "    f'{len(semantic_columns[\"numeric_columns\"])} numeric measures, and '\n"
        "    f'{len(semantic_columns[\"date_columns\"])} date/time fields.'\n"
        ")\n"
        "sample_entities = []\n"
        "if entity_column:\n"
        "    sample_entities = [str(value) for value in df[entity_column].dropna().head(5).tolist()]\n"
        "else:\n"
        "    sample_entities = df.head(5).where(pd.notna(df), None).to_dict(orient='records')\n"
        "dataset_profile = {\n"
        "    'dataset_subject': dataset_subject,\n"
        "    'subject_confidence': subject_confidence,\n"
        "    'subject_reasoning': subject_reasoning,\n"
        "    'primary_entity': primary_entity,\n"
        "    'entity_count': entity_count,\n"
        "    'entity_column': entity_column,\n"
        "    'row_count': row_count,\n"
        "    'column_count': int(len(df.columns)),\n"
        "    'identifier_columns': semantic_columns['identifier_columns'],\n"
        "    'date_columns': semantic_columns['date_columns'],\n"
        "    'numeric_columns': semantic_columns['numeric_columns'],\n"
        "    'categorical_columns': semantic_columns['categorical_columns'] + semantic_columns['boolean_columns'] + semantic_columns['location_columns'],\n"
        "    'text_columns': semantic_columns['text_columns'] + semantic_columns['name_columns'] + semantic_columns['organization_columns'] + semantic_columns['person_columns'],\n"
        "    'semantic_columns': semantic_columns,\n"
        "    'relationships': relationships,\n"
        "    'statistics': statistics,\n"
        "    'sample_entities': sample_entities,\n"
        "    'summary': summary,\n"
        "}\n"
        "result = {\n"
        "    'dataset_profile': dataset_profile,\n"
        "    'row_count': row_count,\n"
        "    'column_count': int(len(df.columns)),\n"
        "    'columns': [str(column) for column in df.columns],\n"
        "    'numeric_columns': [str(column) for column in df.select_dtypes(include='number').columns],\n"
        "    'categorical_columns': [str(column) for column in df.select_dtypes(exclude='number').columns],\n"
        "    'column_profiles': column_profiles,\n"
        "    'sample_records': df.head(5).where(pd.notna(df), None).to_dict(orient='records'),\n"
        "    'example_insights': {\n"
        "        'sample_entities': sample_entities,\n"
        "        'top_grouping_columns': grouping_columns[:5],\n"
        "        'suggested_questions': [\n"
        "            f'How many {primary_entity.lower()} records are there?',\n"
        "            'What information does this dataset contain?',\n"
        "            'Are there missing values?',\n"
        "            'Which categories are most common?',\n"
        "            'Show me examples from the dataset.',\n"
        "        ],\n"
        "    },\n"
        "}\n"
    )


def _chart_code(lower: str, selected: list[str]) -> str:
    if "hist" in lower and selected:
        return f"result = df[[{selected[0]!r}]].dropna()"
    if len(selected) >= 2:
        return f"result = df[[{selected[0]!r}, {selected[1]!r}]].dropna().head(200)"
    return "result = df.select_dtypes(include='number').head(200)"


def _fallback_summary(plan: AnalysisPlan, result: Any, educational_mode: bool) -> str:
    if plan.grounding_status in {"Unsupported", "Not Found"}:
        return "This question cannot be answered from the uploaded dataset. I can only answer questions using information contained in the uploaded data."
    if isinstance(result, dict) and "record_count" in result:
        if "distinct_entity_count" in result and result.get("entity_column"):
            if result["distinct_entity_count"] == result["record_count"]:
                return f"The uploaded dataset contains {result['record_count']:,} records, with {result['distinct_entity_count']:,} unique non-missing values in `{result['entity_column']}`."
            return f"The uploaded dataset contains {result['record_count']:,} rows and {result['distinct_entity_count']:,} unique non-missing values in `{result['entity_column']}`."
        return f"The uploaded dataset contains {result['record_count']:,} records. For this count, each row is treated as one record in the dataset."
    if isinstance(result, dict) and {"column", "metric", "value"}.issubset(result):
        return f"The {result['metric']} of {_display_column(result['column'])} is {round(float(result['value']), 4)}, computed from {result.get('non_null_count', 'the available')} non-missing values."
    if isinstance(result, dict) and isinstance(result.get("dataset_profile"), dict):
        profile = result["dataset_profile"]
        semantic_columns = profile.get("semantic_columns") or {}
        numeric_columns = profile.get("numeric_columns") or []
        categorical_columns = profile.get("categorical_columns") or []
        text_columns = profile.get("text_columns") or []
        sample_entities = profile.get("sample_entities") or []
        examples = ", ".join(str(value) for value in sample_entities[:5])
        column_groups = []
        identifier_display = [column for column in profile.get("identifier_columns", []) if not _is_index_like_column(column)]
        categorical_display = [column for column in categorical_columns if not _is_index_like_column(column)]
        numeric_display = [column for column in numeric_columns if not _is_index_like_column(column)]
        date_display = [column for column in profile.get("date_columns", []) if not _is_index_like_column(column)]
        text_display = [column for column in text_columns if not _is_index_like_column(column)]
        if identifier_display:
            column_groups.append(f"identifiers: {', '.join(_display_column(column) for column in identifier_display[:4])}")
        if categorical_display:
            column_groups.append(f"categories: {', '.join(_display_column(column) for column in categorical_display[:4])}")
        if numeric_display:
            column_groups.append(f"numeric measures: {', '.join(_display_column(column) for column in numeric_display[:6])}")
        if date_display:
            column_groups.append(f"dates/times: {', '.join(_display_column(column) for column in date_display[:4])}")
        if text_display:
            column_groups.append(f"text/name fields: {', '.join(_display_column(column) for column in text_display[:4])}")
        confidence = profile.get("subject_confidence", 0.0)
        parts = [
            profile.get("summary", ""),
            f"Detected subject: {profile.get('dataset_subject')} (confidence {confidence}).",
            f"Primary entity: {profile.get('primary_entity')} with {profile.get('entity_count'):,} estimated unique entities.",
            "Column groups include " + "; ".join(column_groups) + "." if column_groups else "",
            f"Example entities: {examples}." if examples else "",
        ]
        if semantic_columns and educational_mode:
            parts.append("I used the Dataset Profile: semantic column classifications, statistics, relationships, and sample entities.")
        return " ".join(part for part in parts if part)
    if isinstance(result, dict) and {"row_count", "column_count", "columns"}.issubset(result):
        numeric_columns = result.get("numeric_columns") or []
        categorical_columns = result.get("categorical_columns") or []
        meaningful_columns = _meaningful_columns(result)
        column_preview = ", ".join(_display_column(column) for column in meaningful_columns[:8])
        subject = _infer_subject(result)
        profile_sentence = _profile_sentence(result)
        parts = [
            f"This dataset appears to be about {subject}." if subject else "",
            f"It contains {result['row_count']:,} records and {result['column_count']:,} columns.",
            f"Key fields include: {column_preview}." if column_preview else "",
            f"It has {len(numeric_columns)} numeric columns and {len(categorical_columns)} categorical or non-numeric columns.",
            profile_sentence,
        ]
        if educational_mode:
            parts.append("I determined this by inspecting the full dataframe, column profiles, top values, and sample records.")
        return " ".join(part for part in parts if part)
    if isinstance(result, list) and result and isinstance(result[0], dict) and {"value", "count", "percent"}.issubset(result[0]):
        top_rows = result[:5]
        segments = [f"{row['value']}: {row['count']} rows ({row['percent']}%)" for row in top_rows]
        return "Here is the computed breakdown: " + "; ".join(segments) + "."
    if isinstance(result, list) and result and isinstance(result[0], dict) and len(result[0]) == 1:
        column = next(iter(result[0].keys()))
        values = [str(row.get(column)) for row in result[:10] if row.get(column) is not None]
        return f"Here are some {column}: " + ", ".join(values) + "."
    if isinstance(result, dict) and result and all(not isinstance(value, (dict, list)) for value in result.values()):
        if len(result) == 1:
            key, value = next(iter(result.items()))
            return f"The answer is {_display_column(key)}: {value}."
        numeric_items = [(key, value) for key, value in result.items() if isinstance(value, (int, float))]
        if numeric_items:
            top_key, top_value = max(numeric_items, key=lambda item: item[1])
            return f"The top result is {_display_column(top_key)} with {top_value}."
        items = ", ".join(f"{_display_column(key)} = {value}" for key, value in result.items())
        return f"The analysis returned {items}."
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
        row = result[0]
        row_items = ", ".join(f"{_display_column(key)} = {value}" for key, value in row.items())
        return f"The result is {row_items}."
    detail = " The analysis selected the listed columns, ran the planned operations, and returned the displayed result." if educational_mode else ""
    return f"I ran the planned local analysis and returned the result below.{detail}"


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
        unique = profile.get("unique_count")
        top_values = profile.get("top_values") or []
        if top_values:
            top = top_values[0]
            informative.append(f"{_display_column(column)} has {unique} unique values; the most common is {top.get('value')} ({top.get('count')} rows, {top.get('percent')}%).")
        elif profile.get("mean") is not None:
            informative.append(f"{_display_column(column)} is numeric with an average of {round(float(profile.get('mean')), 2)}.")
    return " ".join(informative[:3])


def _meaningful_columns(result: dict[str, Any]) -> list[str]:
    columns = [str(column) for column in result.get("columns", [])]
    non_index = [column for column in columns if not _is_index_like_column(column)]
    return non_index or columns


def _is_index_like_column(column: Any) -> bool:
    text = str(column).strip().lower()
    return text.startswith("unnamed") or text in {"index", "id", "rowid", "row_id"}


def _is_entity_named_column(column: Any) -> bool:
    text = _normalize_text(column)
    return any(term in text.split() or term in text for term in _ENTITY_WORDS | {"name"})


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
    spaced = re.sub(r"[_\\.]+", " ", original).strip()
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", spaced)
    return spaced or original
