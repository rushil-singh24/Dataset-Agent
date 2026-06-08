from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .ollama_agent import build_analysis, summarize_result
from .safe_exec import SafetyError, execute_analysis
from .schemas import ChatRequest, ChatResponse, Confidence, ExecutionResult, TableResult
from .store import ARTIFACT_DIR, state


app = FastAPI(title="DataChat AI", version="0.1.0")
FULL_CONTEXT_MAX_CELLS = int(os.getenv("FULL_CONTEXT_MAX_CELLS", "10000"))
FULL_CONTEXT_MAX_CHARS = int(os.getenv("FULL_CONTEXT_MAX_CHARS", "120000"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _agent_dataset_context(dataset_id: str) -> str:
    dataset = state.get_dataset(dataset_id)
    profile = dataset.profile
    lines = [profile.summary, "", "Column Details:"]
    for column in profile.columns[:40]:
        examples = ", ".join(str(value) for value in column.sample_values[:3])
        detail = (
            f"- {column.name}: dtype={column.dtype}, missing={column.missing_count} "
            f"({column.missing_percent}%), unique={column.unique_count}"
        )
        if examples:
            detail += f", examples={examples}"
        if column.stats:
            stats_preview = ", ".join(f"{key}={value}" for key, value in list(column.stats.items())[:4])
            detail += f", stats={stats_preview}"
        if column.top_values:
            top_preview = ", ".join(f"{item.get('value')} ({item.get('count')})" for item in column.top_values[:3])
            detail += f", top_values={top_preview}"
        lines.append(detail)
    if len(profile.columns) > 40:
        lines.append(f"- {len(profile.columns) - 40} additional columns omitted from prompt context.")
    cell_count = len(dataset.df) * max(len(dataset.df.columns), 1)
    if cell_count <= FULL_CONTEXT_MAX_CELLS:
        records = dataset.df.where(dataset.df.notna(), None).to_dict(orient="records")
        records_json = json.dumps(records, default=str)
        if len(records_json) <= FULL_CONTEXT_MAX_CHARS:
            lines.extend(["", "Full Dataset Records:", records_json])
        else:
            lines.extend(
                [
                    "",
                    "Full Dataset Records: omitted from model prompt because serialized data exceeds context budget.",
                    "The complete dataframe is still loaded as df for generated Pandas/DuckDB execution over all rows and columns.",
                ]
            )
    else:
        lines.extend(
            [
                "",
                f"Full Dataset Records: omitted from model prompt because dataset has {cell_count:,} cells.",
                "The complete dataframe is still loaded as df for generated Pandas/DuckDB execution over all rows and columns.",
            ]
        )
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
        record = state.save_upload(file.filename, content)
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


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        dataset = state.get_dataset(request.dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    session = state.get_session(request.session_id)
    steps = [
        {"label": "Understanding Question", "state": "complete"},
        {"label": "Building Analysis Plan", "state": "complete"},
    ]
    analysis = await build_analysis(
        request.message,
        _agent_dataset_context(request.dataset_id),
        dataset.profile.metadata.columns,
        session.messages + session.previous_results,
    )

    tables: list[TableResult] = []
    charts = []
    raw_result = None
    generated_code = analysis.code

    if analysis.plan.grounding_status == "Needs Clarification":
        steps.append({"label": "Grounding Verification", "state": "blocked"})
        response = ChatResponse(
            assistant_message=analysis.plan.column_selection_reason,
            classification=analysis.classification,
            analysis_plan=analysis.plan,
            execution=ExecutionResult(status_steps=steps, generated_code=None),
            confidence=Confidence(level="Low", reason="The dataset is loaded, but the request needs a more specific column or field."),
        )
    elif analysis.classification == "Unsupported Request" or analysis.plan.grounding_status in {"Unsupported", "Not Found"}:
        steps.append({"label": "Grounding Verification", "state": "blocked"})
        response = ChatResponse(
            assistant_message="This question cannot be answered from the uploaded dataset. I can only answer questions using information contained in the uploaded data.",
            classification=analysis.classification,
            analysis_plan=analysis.plan,
            execution=ExecutionResult(status_steps=steps, generated_code=None),
            confidence=Confidence(level="High", reason="The request is outside the uploaded dataset or required fields were not found."),
        )
    else:
        try:
            steps.extend(
                [
                    {"label": "Generating Code", "state": "complete"},
                    {"label": "Running Analysis", "state": "running"},
                ]
            )
            if generated_code:
                raw_result, tables, charts = execute_analysis(generated_code, dataset.df, analysis.chart_type)
            steps[-1]["state"] = "complete"
            if analysis.chart_type != "none":
                steps.append({"label": "Generating Visualization", "state": "complete" if charts else "skipped"})
            steps.append({"label": "Summarizing Results", "state": "complete"})

            missing_relevant_values = any(
                column.name in analysis.plan.selected_columns and column.missing_count > 0
                for column in dataset.profile.columns
            )
            summary = await summarize_result(
                request.message,
                analysis.plan,
                raw_result,
                request.educational_mode,
                missing_relevant_values,
            )
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
        except SafetyError as exc:
            steps.append({"label": "Safety Validation", "state": "blocked"})
            response = ChatResponse(
                assistant_message=f"I could not safely run the generated analysis: {exc}",
                classification=analysis.classification,
                analysis_plan=analysis.plan,
                execution=ExecutionResult(status_steps=steps, generated_code=generated_code),
                confidence=Confidence(level="Low", reason="The generated code failed safety validation or execution."),
            )

    session.messages.append({"role": "user", "content": request.message})
    session.messages.append({"role": "assistant", "content": response.assistant_message})
    session.previous_results.append(
        {
            "question": request.message,
            "classification": response.classification,
            "plan": response.analysis_plan.model_dump(mode="json"),
            "result": response.execution.raw_result,
        }
    )
    session.messages[:] = session.messages[-12:]
    session.previous_results[:] = session.previous_results[-6:]
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
