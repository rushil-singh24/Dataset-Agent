"""FastAPI application for DataChat AI."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .ollama_agent import build_analysis, build_analysis_with_code_retry, summarize_result
from .safe_exec import SafetyError, execute_analysis
from .schemas import AnalysisPlan, ChatRequest, ChatResponse, Confidence, ExecutionResult, TableResult
from .store import ARTIFACT_DIR, state

app = FastAPI(title="DataChat AI", version="0.2.2")

FULL_CONTEXT_MAX_CELLS = int(os.getenv("FULL_CONTEXT_MAX_CELLS", "10000"))
FULL_CONTEXT_MAX_CHARS = int(os.getenv("FULL_CONTEXT_MAX_CHARS", "120000"))
MAX_EXEC_RETRIES = int(os.getenv("MAX_EXEC_RETRIES", "3"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
MAX_DATASET_ROWS = int(os.getenv("MAX_DATASET_ROWS", "50000"))
MAX_DATASET_COLUMNS = int(os.getenv("MAX_DATASET_COLUMNS", "500"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def reset_runtime_state_on_startup() -> None:
    """Make each local app run start fresh.

    Datasets and chat history are intentionally runtime-only. Uploaded files and
    generated artifacts from earlier runs are deleted so the app remains generic
    for whoever starts it locally.
    """
    state.reset_runtime(clear_files=True)


def _agent_dataset_context(dataset_id: str) -> str:
    """Build a compact, model-friendly context block for the uploaded dataset.

    Keep this factual and generic. Avoid domain assumptions because the user can upload
    any dataset.
    """
    dataset = state.get_dataset(dataset_id)
    profile = dataset.profile
    lines = [profile.summary, "", "Column Details:"]

    for column in profile.columns[:40]:
        examples = ", ".join(str(v) for v in column.sample_values[:3])
        detail = (
            f"- {column.name}: dtype={column.dtype}, "
            f"missing={column.missing_count} ({column.missing_percent}%), "
            f"unique={column.unique_count}"
        )
        if examples:
            detail += f", examples={examples}"
        if column.stats:
            stats_preview = ", ".join(f"{k}={v}" for k, v in list(column.stats.items())[:4])
            detail += f", stats={stats_preview}"
        if column.top_values:
            top_preview = ", ".join(f"{item.get('value')} ({item.get('count')})" for item in column.top_values[:3])
            detail += f", top_values={top_preview}"
        lines.append(detail)

    if len(profile.columns) > 40:
        lines.append(f"- {len(profile.columns) - 40} additional columns omitted from prompt.")

    cell_count = len(dataset.df) * max(len(dataset.df.columns), 1)
    if cell_count <= FULL_CONTEXT_MAX_CELLS:
        records = dataset.df.where(dataset.df.notna(), None).head(250).to_dict(orient="records")
        records_json = json.dumps(records, default=str)
        if len(records_json) <= FULL_CONTEXT_MAX_CHARS:
            lines.extend(["", "Dataset Sample Records:", records_json])
        else:
            lines.extend([
                "",
                "Dataset Sample Records: omitted because serialized data exceeds context budget.",
                "The complete dataframe is loaded as `df` for code execution.",
            ])
    else:
        sample = dataset.df.where(dataset.df.notna(), None).head(20).to_dict(orient="records")
        lines.extend([
            "",
            f"Full Dataset Records: omitted because dataset has {cell_count:,} cells.",
            "Small sample records:",
            json.dumps(sample, default=str),
            "The complete dataframe is loaded as `df` for code execution.",
        ])

    return "\n".join(lines)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if not file.filename:
            raise ValueError("Missing filename.")
        if len(content) > MAX_UPLOAD_BYTES:
            max_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
            raise ValueError(f"File is too large. Maximum upload size is {max_mb:.0f} MB.")

        record = state.save_upload(file.filename, content)
        row_count = len(record.df)
        column_count = len(record.df.columns)
        if row_count > MAX_DATASET_ROWS or column_count > MAX_DATASET_COLUMNS:
            dataset_id = record.dataset_id
            state.datasets.pop(dataset_id, None)
            try:
                record.stored_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise ValueError(
                f"Dataset is too large for this app. Maximum is {MAX_DATASET_ROWS:,} rows "
                f"and {MAX_DATASET_COLUMNS:,} columns. Uploaded dataset has {row_count:,} rows "
                f"and {column_count:,} columns."
            )
        return record.profile.model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dataset upload failed: {exc}") from exc


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    try:
        return state.get_dataset(dataset_id).profile.model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/sessions/{session_id}/clear")
def clear_session(session_id: str):
    state.clear_session(session_id)
    return {"status": "cleared"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        dataset = state.get_dataset(request.dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    session = state.get_session(request.session_id)

    fast_response = _try_fast_path_response(request.message, dataset, request.educational_mode)
    if fast_response is not None:
        _update_session(session, request.message, fast_response)
        return fast_response

    dataset_context = _agent_dataset_context(request.dataset_id)
    steps = [{"label": "Understanding Question", "state": "running"}]

    analysis = await build_analysis(
        request.message,
        dataset_context,
        dataset.profile.metadata.columns,
        session.messages + session.previous_results,
    )
    steps[0]["state"] = "complete"
    steps.append({"label": "Building Analysis Plan", "state": "complete"})

    if (
        analysis.classification == "Unsupported Request"
        or analysis.plan.grounding_status in {"Unsupported", "Not Found", "Needs Clarification"}
        or not analysis.code
    ):
        if not analysis.code and analysis.plan.grounding_status == "Ready":
            user_message = (
                "I understood the question, but the analysis agent did not return executable code. "
                "Try asking this as a direct dataset question, or include the exact column name you want analyzed."
            )
        else:
            user_message = analysis.plan.column_selection_reason or (
                "I could not answer that from the uploaded dataset. Please ask a question about the data."
            )
        steps.append({"label": "Grounding Verification", "state": "blocked"})
        response = _build_response(
            message=user_message,
            analysis=analysis,
            steps=steps,
            confidence=Confidence(level="Low", reason="Question could not be answered from the uploaded dataset."),
        )
        _update_session(session, request.message, response)
        return response

    steps.append({"label": "Generating Code", "state": "complete"})
    running_index = len(steps)
    steps.append({"label": "Running Analysis", "state": "running"})

    tables: list[TableResult] = []
    charts = []
    raw_result = None
    generated_code = analysis.code
    exec_error: str | None = None

    for exec_attempt in range(1, MAX_EXEC_RETRIES + 1):
        if not generated_code:
            exec_error = "No code was generated."
            break

        try:
            raw_result, tables, charts = execute_analysis(generated_code, dataset.df, analysis.chart_type)
            exec_error = None
            break
        except SafetyError as exc:
            exec_error = f"Safety validation error: {exc}"
        except Exception as exc:  # noqa: BLE001
            exec_error = f"Runtime error: {exc}"

        if exec_attempt < MAX_EXEC_RETRIES:
            fix_analysis = await build_analysis_with_code_retry(
                message=request.message,
                dataset_context=dataset_context,
                columns=dataset.profile.metadata.columns,
                history=session.messages + session.previous_results,
                execution_error=exec_error,
                previous_code=generated_code,
            )
            generated_code = fix_analysis.code
            analysis = fix_analysis

    if exec_error is not None:
        steps[running_index]["state"] = "blocked"
        steps.append({"label": "Code Fix Attempts Exhausted", "state": "blocked"})
        response = _build_response(
            message=(
                f"I tried {MAX_EXEC_RETRIES} times but could not produce working code for this question. "
                f"Last error: {exec_error}"
            ),
            analysis=analysis,
            steps=steps,
            generated_code=generated_code,
            confidence=Confidence(level="Low", reason="Code execution failed after all retry attempts."),
        )
        _update_session(session, request.message, response)
        return response

    steps[running_index]["state"] = "complete"

    if analysis.chart_type != "none":
        steps.append({"label": "Generating Visualization", "state": "complete" if charts else "skipped"})

    steps.append({"label": "Summarizing Results", "state": "running"})
    summary_index = len(steps) - 1

    missing_relevant_values = any(
        col.name in analysis.plan.selected_columns and col.missing_count > 0
        for col in dataset.profile.columns
    )

    summary = await summarize_result(
        request.message,
        analysis.plan,
        raw_result,
        request.educational_mode,
        missing_relevant_values,
    )
    steps[summary_index]["state"] = "complete"
    steps.append({"label": "Complete", "state": "complete"})

    response = ChatResponse(
        assistant_message=summary.answer,
        classification=analysis.classification,
        analysis_plan=analysis.plan,
        execution=ExecutionResult(
            status_steps=steps,
            generated_code=generated_code,
            tables=tables,
            charts=charts,
            raw_result=raw_result,
        ),
        confidence=summary.confidence,
    )
    _update_session(session, request.message, response)
    return response


@app.get("/api/artifacts/{artifact_id}")
def get_artifact(artifact_id: str):
    artifact_root = ARTIFACT_DIR.resolve()
    path = (ARTIFACT_DIR / artifact_id).resolve()
    if not path.is_relative_to(artifact_root) or not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(path)


@app.post("/api/tables/export")
def export_table(payload: dict):
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise HTTPException(status_code=400, detail="Expected rows list.")
    from uuid import uuid4

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_id = f"{uuid4().hex}.csv"
    path = ARTIFACT_DIR / artifact_id
    pd.DataFrame(rows).to_csv(path, index=False)
    return {"artifact_id": artifact_id, "url": f"/api/artifacts/{artifact_id}"}


def _try_fast_path_response(message: str, dataset: Any, educational_mode: bool) -> ChatResponse | None:
    """Answer simple dataset questions without using Ollama or executing generated code.

    This prevents simple prompts from displaying "Building plan / generating code / running analysis"
    and improves accuracy for basic metadata answers.
    """
    lower = message.lower().strip()
    df = dataset.df
    profile = dataset.profile

    if _is_complex_question(lower):
        return None

    steps = [
        {"label": "Inspecting Dataset", "state": "complete"},
        {"label": "Answering from Cached Metadata", "state": "complete"},
        {"label": "Complete", "state": "complete"},
    ]

    if _asks_overview(lower):
        numeric_count = int(len(df.select_dtypes(include="number").columns))
        categorical_count = int(len(df.columns) - numeric_count)
        missing_total = int(df.isna().sum().sum())
        column_preview = ", ".join(str(col) for col in df.columns[:12])
        if len(df.columns) > 12:
            column_preview += f", and {len(df.columns) - 12} more"
        raw_result = {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "numeric_columns": numeric_count,
            "non_numeric_columns": categorical_count,
            "missing_values": missing_total,
            "columns_preview": column_preview,
        }
        answer = (
            f"This dataset has {len(df):,} rows and {len(df.columns):,} columns. "
            f"It includes {numeric_count:,} numeric columns and {categorical_count:,} non-numeric columns. "
            f"There are {missing_total:,} missing values overall. "
            f"Some columns include: {column_preview}."
        )
        return _direct_response(message, answer, "Dataset Information", [], steps, raw_result, educational_mode)

    if _asks_shape(lower):
        raw_result = {"row_count": int(len(df)), "column_count": int(len(df.columns))}
        answer = f"This dataset has {len(df):,} rows and {len(df.columns):,} columns."
        return _direct_response(message, answer, "Dataset Information", [], steps, raw_result, educational_mode)

    if _asks_columns(lower):
        columns = [str(col) for col in df.columns]
        raw_result = {"columns": columns}
        preview = ", ".join(columns[:30])
        if len(columns) > 30:
            preview += f", and {len(columns) - 30} more"
        answer = f"The dataset columns are: {preview}."
        return _direct_response(message, answer, "Dataset Information", [], steps, raw_result, educational_mode)

    if _asks_schema(lower):
        schema = pd.DataFrame({
            "column": df.columns.astype(str),
            "dtype": df.dtypes.astype(str).values,
            "missing_count": df.isna().sum().astype(int).values,
            "unique_count": df.nunique(dropna=True).astype(int).values,
        })
        answer = f"The dataset has {len(df.columns):,} columns. I included each column's data type, missing count, and unique count."
        return _direct_response(
            message,
            answer,
            "Dataset Information",
            [],
            steps,
            schema,
            educational_mode,
            tables=[_table_from_dataframe(schema, "Schema")],
        )

    if _asks_missing(lower):
        missing = pd.DataFrame({
            "column": df.columns.astype(str),
            "missing_count": df.isna().sum().astype(int).values,
            "missing_percent": (df.isna().mean().values * 100).round(2),
        }).sort_values("missing_count", ascending=False)
        total_missing = int(df.isna().sum().sum())
        answer = f"There are {total_missing:,} missing values across the dataset. I included the missing count and percentage for each column."
        return _direct_response(
            message,
            answer,
            "Data Quality Inspection",
            [],
            steps,
            missing,
            educational_mode,
            tables=[_table_from_dataframe(missing, "Missing Values")],
        )

    if _asks_sample(lower):
        limit = _extract_limit(lower)
        sample = df.head(limit)
        answer = f"Here are the first {min(limit, len(df)):,} rows of the dataset."
        return _direct_response(
            message,
            answer,
            "Dataset Information",
            [],
            steps,
            sample,
            educational_mode,
            tables=[_table_from_dataframe(sample, "Sample Rows")],
        )

    selected_columns = _select_columns(lower, list(df.columns))

    if _asks_basic_stat(lower) and selected_columns:
        numeric_col = _first_numeric_column(df, selected_columns)
        if numeric_col is None:
            return None
        series = pd.to_numeric(df[numeric_col], errors="coerce")
        stat_name, value = _compute_requested_stat(lower, series)
        if value is None:
            return None
        raw_result = {"column": numeric_col, stat_name: value}
        answer = f"The {stat_name.replace('_', ' ')} of `{numeric_col}` is {value:,.4g}."
        return _direct_response(message, answer, "Statistical Analysis", [numeric_col], steps, raw_result, educational_mode)

    if _asks_value_counts(lower) and selected_columns:
        col = selected_columns[-1]
        counts = df[col].dropna().astype(str).value_counts().head(25)
        result = pd.DataFrame({
            "value": counts.index.astype(str),
            "count": counts.values.astype(int),
            "percent": (counts.values / max(len(df), 1) * 100).round(2),
        })
        answer = f"Here are the most common values in `{col}`."
        return _direct_response(
            message,
            answer,
            "Statistical Analysis",
            [col],
            steps,
            result,
            educational_mode,
            tables=[_table_from_dataframe(result, f"Value Counts: {col}")],
        )

    if _asks_numeric_summary(lower):
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return None
        summary = numeric.describe().round(4).transpose().reset_index().rename(columns={"index": "column"})
        answer = "Here are descriptive statistics for the numeric columns in the dataset."
        return _direct_response(
            message,
            answer,
            "Statistical Analysis",
            list(numeric.columns),
            steps,
            summary,
            educational_mode,
            tables=[_table_from_dataframe(summary, "Numeric Summary")],
        )

    return None


def _direct_response(
    message: str,
    answer: str,
    category: str,
    selected_columns: list[str],
    steps: list[dict[str, str]],
    raw_result: Any,
    educational_mode: bool,
    tables: list[TableResult] | None = None,
) -> ChatResponse:
    if educational_mode:
        answer += " I answered this directly from the dataset metadata or dataframe, so no LLM code-generation step was needed."

    plan = AnalysisPlan(
        category=category,  # type: ignore[arg-type]
        question_understanding=message,
        selected_columns=selected_columns,
        column_selection_reason="Answered directly from cached dataset metadata/dataframe without generated code.",
        planned_operations=["Use deterministic backend shortcut for simple dataset question"],
        output_type="Text + Table" if tables else "Text",
        estimated_complexity="Low",
        grounding_status="Ready",
        status="No Execution Needed",
    )
    return ChatResponse(
        assistant_message=answer,
        classification=category,  # type: ignore[arg-type]
        analysis_plan=plan,
        execution=ExecutionResult(
            status_steps=steps,
            generated_code=None,
            tables=tables or [],
            charts=[],
            raw_result=_json_safe(raw_result),
        ),
        confidence=Confidence(level="High", reason="Answered deterministically from the uploaded dataset."),
    )


def _build_response(
    *,
    message: str,
    analysis,
    steps: list,
    generated_code: str | None = None,
    tables: list | None = None,
    charts: list | None = None,
    raw_result=None,
    confidence: Confidence,
) -> ChatResponse:
    return ChatResponse(
        assistant_message=message,
        classification=analysis.classification,
        analysis_plan=analysis.plan,
        execution=ExecutionResult(
            status_steps=steps,
            generated_code=generated_code,
            tables=tables or [],
            charts=charts or [],
            raw_result=raw_result,
        ),
        confidence=confidence,
    )


def _update_session(session, user_message: str, response: ChatResponse) -> None:
    session.messages.append({"role": "user", "content": user_message})
    session.messages.append({"role": "assistant", "content": response.assistant_message})
    session.previous_results.append({
        "question": user_message,
        "classification": response.classification,
        "plan": response.analysis_plan.model_dump(mode="json"),
        "result": response.execution.raw_result,
    })
    session.messages[:] = session.messages[-20:]
    session.previous_results[:] = session.previous_results[-6:]


def _is_complex_question(lower: str) -> bool:
    complex_terms = [
        "plot", "chart", "graph", "visual", "visualize", "correlation", "correlate", "relationship",
        "regression", "predict", "model", "cluster", "trend", "over time", "compare", "versus", "vs",
        "by ", "group", "grouped", "segment", "why", "explain why", "insight", "recommend", "outlier",
    ]
    return any(term in lower for term in complex_terms)


def _asks_overview(lower: str) -> bool:
    return any(term in lower for term in [
        "overview", "summarize dataset", "summary of dataset", "tell me about",
        "what is this dataset", "what's this dataset", "describe dataset", "basic info",
    ])


def _asks_shape(lower: str) -> bool:
    return any(term in lower for term in [
        "how many rows", "row count", "number of rows", "how many records", "record count",
        "shape", "dimension", "dimensions", "how big", "how many columns",
    ])


def _asks_columns(lower: str) -> bool:
    return any(term in lower for term in ["what columns", "column names", "list columns", "show columns", "fields"])


def _asks_schema(lower: str) -> bool:
    return any(term in lower for term in ["schema", "dtypes", "data types", "types of columns", "column types"])


def _asks_missing(lower: str) -> bool:
    return any(term in lower for term in ["missing", "null", "nan", "blank", "completeness", "data quality"])


def _asks_sample(lower: str) -> bool:
    return any(term in lower for term in ["sample", "examples", "show rows", "first rows", "preview", "head"])


def _asks_basic_stat(lower: str) -> bool:
    return any(term in lower for term in ["average", "mean", "median", "minimum", "maximum", " min ", " max ", "sum", "total"])


def _asks_numeric_summary(lower: str) -> bool:
    return any(term in lower for term in ["summary stats", "summary statistics", "describe", "descriptive statistics", "statistics"])


def _asks_value_counts(lower: str) -> bool:
    return any(term in lower for term in ["value counts", "distribution", "breakdown", "frequency", "most common", "categories", "percentage", "percent"])


def _extract_limit(lower: str) -> int:
    match = re.search(r"\b(\d{1,2})\b", lower)
    if match:
        return max(1, min(int(match.group(1)), 25))
    return 10


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


def _first_numeric_column(df: pd.DataFrame, selected_columns: list[str]) -> str | None:
    for col in selected_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return col
    for col in selected_columns:
        if col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any():
                return col
    return None


def _compute_requested_stat(lower: str, series: pd.Series) -> tuple[str, float | None]:
    clean = series.dropna()
    if clean.empty:
        return "value", None
    if "median" in lower:
        return "median", float(clean.median())
    if "minimum" in lower or re.search(r"\bmin\b", lower):
        return "minimum", float(clean.min())
    if "maximum" in lower or re.search(r"\bmax\b", lower):
        return "maximum", float(clean.max())
    if "sum" in lower or "total" in lower:
        return "sum", float(clean.sum())
    return "average", float(clean.mean())


def _table_from_dataframe(df: pd.DataFrame, title: str) -> TableResult:
    return TableResult(
        title=title,
        columns=[str(col) for col in df.columns],
        rows=df.where(pd.notna(df), None).head(100).to_dict(orient="records"),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.where(pd.notna(value), None).head(100).to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.where(pd.notna(value), None).head(100).to_dict()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
