from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from .schemas import ColumnProfile, DatasetMetadata, DatasetProfile


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def profile_dataframe(dataset_id: str, name: str, df: pd.DataFrame, uploaded_at: datetime) -> DatasetProfile:
    memory_usage_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))
    metadata = DatasetMetadata(
        dataset_id=dataset_id,
        name=name,
        uploaded_at=uploaded_at,
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        columns=[str(column) for column in df.columns],
        dtypes={str(column): str(dtype) for column, dtype in df.dtypes.items()},
        memory_usage_mb=round(memory_usage_mb, 3),
    )

    profiles: list[ColumnProfile] = []
    for column in df.columns:
        series = df[column]
        missing_count = int(series.isna().sum())
        missing_percent = round((missing_count / max(len(df), 1)) * 100, 2)
        sample_values = [_json_safe(value) for value in series.dropna().head(5).tolist()]
        stats: dict[str, Any] = {}
        top_values: list[dict[str, Any]] = []

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe(percentiles=[0.25, 0.5, 0.75])
            stats = {
                "mean": _json_safe(desc.get("mean")),
                "median": _json_safe(series.median()),
                "std": _json_safe(desc.get("std")),
                "min": _json_safe(desc.get("min")),
                "max": _json_safe(desc.get("max")),
                "q1": _json_safe(desc.get("25%")),
                "q3": _json_safe(desc.get("75%")),
            }
        else:
            counts = series.dropna().astype(str).value_counts().head(10)
            top_values = [
                {"value": _json_safe(index), "count": int(count)}
                for index, count in counts.items()
            ]

        profiles.append(
            ColumnProfile(
                name=str(column),
                dtype=str(series.dtype),
                missing_count=missing_count,
                missing_percent=missing_percent,
                unique_count=int(series.nunique(dropna=True)),
                sample_values=sample_values,
                stats=stats,
                top_values=top_values,
            )
        )

    empty_columns = [str(column) for column in df.columns if df[column].isna().all()]
    constant_columns = [str(column) for column in df.columns if df[column].nunique(dropna=False) <= 1]
    high_cardinality_columns = [
        str(column)
        for column in df.columns
        if len(df) > 0 and df[column].nunique(dropna=True) / len(df) > 0.8 and df[column].nunique(dropna=True) > 50
    ]

    numeric_columns = [str(column) for column in df.select_dtypes(include="number").columns]
    categorical_columns = [str(column) for column in df.select_dtypes(exclude="number").columns]
    likely_targets = [
        column
        for column in metadata.columns
        if column.lower() in {"target", "label", "churn", "outcome", "class", "y"}
    ]

    summary = "\n".join(
        [
            f"Dataset Name: {name}",
            f"Rows: {metadata.row_count}",
            f"Columns: {metadata.column_count}",
            f"Target Variables: {', '.join(likely_targets) if likely_targets else 'None detected'}",
            f"Numerical Features: {', '.join(numeric_columns[:20]) if numeric_columns else 'None'}",
            f"Categorical Features: {', '.join(categorical_columns[:20]) if categorical_columns else 'None'}",
            "Potential Business Use: Exploratory analysis, data quality review, trend analysis, and reporting based on uploaded data.",
        ]
    )

    return DatasetProfile(
        metadata=metadata,
        columns=profiles,
        duplicate_rows=int(df.duplicated().sum()),
        empty_columns=empty_columns,
        constant_columns=constant_columns,
        high_cardinality_columns=high_cardinality_columns,
        summary=summary,
    )
