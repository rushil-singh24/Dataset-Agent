"""
main.py  –  FastAPI application

The analysis pipeline is now fully LLM-driven:
  1. build_analysis()   – LLM produces plan + pandas code
  2. execute_analysis() – safe sandbox runs the code
  3. If execution fails, call build_analysis_with_code_retry() and try again
  4. summarize_result() – LLM explains the result in plain English

No keyword routing, no hardcoded Pandas snippets.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .ollama_agent import build_analysis, build_analysis_with_code_retry, summarize_result
from .safe_exec import SafetyError, execute_analysis
from .schemas import (
    ChatRequest,
    ChatResponse,
    Confidence,
    ExecutionResult,
    TableResult,
)
from .store import ARTIFACT_DIR, state


app = FastAPI(title="DataChat AI", version="0.2.0")

# Maximum cells before we drop the full-records block from the LLM prompt
FULL_CONTEXT_MAX_CELLS = int(os.getenv("FULL_CONTEXT_MAX_CELLS", "10000"))
FULL_CONTEXT_MAX_CHARS = int(os.getenv("FULL_CONTEXT_MAX_CHARS", "120000"))

# How many execution errors to retry through the LLM before giving up
MAX_EXEC_RETRIES = int(os.getenv("MAX_EXEC_RETRIES", "3"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dataset context builder (unchanged from original – works well)
# ---------------------------------------------------------------------------

def _agent_dataset_context(dataset_id: str) -> str:
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
            stats_preview = ", ".join(
                f"{k}={v}" for k, v in list(column.stats.items())[:4]
            )
            detail += f", stats={stats_preview}"
        if column.top_values:
            top_preview = ", ".join(
                f"{item.get('value')} ({item.get('count')})"
                for item in column.top_values[:3]
            )
            detail += f", top_values={top_preview}"
        lines.append(detail)

    if len(profile.columns) > 40:
        lines.append(
            f"- {len(profile.columns) - 40} additional columns omitted from prompt."
        )

    cell_count = len(dataset.df) * max(len(dataset.df.columns), 1)
    if cell_count <= FULL_CONTEXT_MAX_CELLS:
        records = dataset.df.where(dataset.df.notna(), None).to_dict(orient="records")
        records_json = json.dumps(records, default=str)
        if len(records_json) <= FULL_CONTEXT_MAX_CHARS:
            lines.extend(["", "Full Dataset Records:", records_json])
        else:
            lines.extend([
                "",
                "Full Dataset Records: omitted (serialised data exceeds context budget).",
                "The complete dataframe is loaded as `df` for code execution.",
            ])
    else:
        lines.extend([
            "",
            f"Full Dataset Records: omitted (dataset has {cell_count:,} cells).",
            "The complete dataframe is loaded as `df` for code execution.",
        ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if not file.filename:
            raise ValueError("Missing filename.")
        record = state.save_upload(file.filename, content)
        return record.profile.model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Dataset upload failed: {exc}"
        ) from exc


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    try:
        return state.get_dataset(dataset_id).profile.model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    # ── 1. Load dataset & session ──────────────────────────────────────────
    try:
        dataset = state.get_dataset(request.dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    session = state.get_session(request.session_id)
    dataset_context = _agent_dataset_context(request.dataset_id)

    # ── 2. Build initial status steps ─────────────────────────────────────
    steps = [
        {"label": "Understanding Question", "state": "running"},
    ]

    # ── 3. Ask LLM for analysis plan + code ───────────────────────────────
    analysis = await build_analysis(
        request.message,
        dataset_context,
        dataset.profile.metadata.columns,
        session.messages + session.previous_results,
    )
    steps[0]["state"] = "complete"
    steps.append({"label": "Building Analysis Plan", "state": "complete"})

    # ── 4. Handle unanswerble / clarification cases ───────────────────────
    if analysis.classification == "Unsupported Request" or analysis.plan.grounding_status in {
        "Unsupported",
        "Not Found",
        "Needs Clarification",
    }:
        # Use the model's own explanation as the message
        user_message = analysis.plan.column_selection_reason or (
            "I couldn't answer that from the uploaded dataset. "
            "Please ask a question about the data."
        )
        steps.append({"label": "Grounding Verification", "state": "blocked"})
        response = _build_response(
            message=user_message,
            analysis=analysis,
            steps=steps,
            confidence=Confidence(
                level="Low",
                reason="Question could not be answered from the uploaded dataset.",
            ),
        )
        _update_session(session, request.message, response)
        return response

    # ── 5. Execute code, with LLM-guided retry on failure ─────────────────
    steps.append({"label": "Generating Code", "state": "complete"})
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
            raw_result, tables, charts = execute_analysis(
                generated_code, dataset.df, analysis.chart_type
            )
            exec_error = None
            break  # success

        except SafetyError as exc:
            exec_error = f"Safety validation error: {exc}"
        except Exception as exc:  # noqa: BLE001
            exec_error = f"Runtime error: {exc}"

        if exec_attempt < MAX_EXEC_RETRIES:
            # Ask the LLM to fix its own code
            steps[-1]["state"] = "running"  # keep "Running Analysis" active
            fix_analysis = await build_analysis_with_code_retry(
                message=request.message,
                dataset_context=dataset_context,
                columns=dataset.profile.metadata.columns,
                history=session.messages + session.previous_results,
                execution_error=exec_error,
                previous_code=generated_code,
            )
            generated_code = fix_analysis.code
            # Propagate updated plan/chart info from the fix attempt
            analysis = fix_analysis

    # ── 6. Handle persistent execution failure ────────────────────────────
    if exec_error is not None:
        steps[-1]["state"] = "blocked"
        steps.append({"label": "Code Fix Attempts Exhausted", "state": "blocked"})
        response = _build_response(
            message=(
                f"I tried {MAX_EXEC_RETRIES} times but could not produce working code "
                f"for this question. Last error: {exec_error}"
            ),
            analysis=analysis,
            steps=steps,
            generated_code=generated_code,
            confidence=Confidence(
                level="Low",
                reason="Code execution failed after all retry attempts.",
            ),
        )
        _update_session(session, request.message, response)
        return response

    steps[-1]["state"] = "complete"

    # ── 7. Visualization step ──────────────────────────────────────────────
    if analysis.chart_type != "none":
        steps.append({
            "label": "Generating Visualization",
            "state": "complete" if charts else "skipped",
        })

    # ── 8. Summarise result via LLM ────────────────────────────────────────
    steps.append({"label": "Summarizing Results", "state": "running"})

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
    steps[-1]["state"] = "complete"
    steps.append({"label": "Complete", "state": "complete"})

    # ── 9. Build & return final response ──────────────────────────────────
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
    path = (ARTIFACT_DIR / artifact_id).resolve()
    if not str(path).startswith(str(ARTIFACT_DIR.resolve())) or not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(path)


@app.post("/api/tables/export")
def export_table(payload: dict):
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise HTTPException(status_code=400, detail="Expected rows list.")
    import pandas as pd
    from uuid import uuid4

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_id = f"{uuid4().hex}.csv"
    path = ARTIFACT_DIR / artifact_id
    pd.DataFrame(rows).to_csv(path, index=False)
    return {"artifact_id": artifact_id, "url": f"/api/artifacts/{artifact_id}"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    session.messages.append(
        {"role": "assistant", "content": response.assistant_message}
    )
    session.previous_results.append({
        "question": user_message,
        "classification": response.classification,
        "plan": response.analysis_plan.model_dump(mode="json"),
        "result": response.execution.raw_result,
    })
    # Keep memory bounded
    session.messages[:] = session.messages[-20:]
    session.previous_results[:] = session.previous_results[-6:]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)