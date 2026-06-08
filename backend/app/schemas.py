from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


Classification = Literal[
    "Dataset Information",
    "Statistical Analysis",
    "Correlation Analysis",
    "Visualization Request",
    "Data Quality Inspection",
    "Trend Analysis",
    "Aggregation Request",
    "Unsupported Request",
]


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_percent: float
    unique_count: int
    sample_values: list[Any] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    top_values: list[dict[str, Any]] = Field(default_factory=list)


class DatasetMetadata(BaseModel):
    dataset_id: str
    name: str
    uploaded_at: datetime
    row_count: int
    column_count: int
    columns: list[str]
    dtypes: dict[str, str]
    memory_usage_mb: float


class DatasetProfile(BaseModel):
    metadata: DatasetMetadata
    columns: list[ColumnProfile]
    duplicate_rows: int
    empty_columns: list[str]
    constant_columns: list[str]
    high_cardinality_columns: list[str]
    summary: str


class AnalysisPlan(BaseModel):
    category: Classification
    question_understanding: str
    selected_columns: list[str] = Field(default_factory=list)
    column_selection_reason: str
    planned_operations: list[str] = Field(default_factory=list)
    output_type: str
    estimated_complexity: Literal["Low", "Medium", "High"] = "Low"
    grounding_status: Literal["Ready", "Needs Clarification", "Not Found", "Unsupported"] = "Ready"
    status: str = "Ready to Execute"


class Confidence(BaseModel):
    level: Literal["High", "Medium", "Low"]
    reason: str


class TableResult(BaseModel):
    title: str
    columns: list[str]
    rows: list[dict[str, Any]]


class ChartResult(BaseModel):
    artifact_id: str
    title: str
    type: str
    url: str


class ExecutionResult(BaseModel):
    status_steps: list[dict[str, Any]]
    generated_code: str | None = None
    tables: list[TableResult] = Field(default_factory=list)
    charts: list[ChartResult] = Field(default_factory=list)
    raw_result: Any = None


class ChatRequest(BaseModel):
    dataset_id: str
    message: str
    session_id: str = "default"
    educational_mode: bool = False


class ChatResponse(BaseModel):
    assistant_message: str
    classification: Classification
    analysis_plan: AnalysisPlan
    execution: ExecutionResult
    confidence: Confidence


class GeneratedAnalysis(BaseModel):
    classification: Classification
    plan: AnalysisPlan
    code: str | None = None
    chart_type: Literal["line", "bar", "scatter", "histogram", "heatmap", "none"] = "none"


class SummaryResponse(BaseModel):
    answer: str
    confidence: Confidence
