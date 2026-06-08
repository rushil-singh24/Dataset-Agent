from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from .loader import load_dataframe
from .profiling import profile_dataframe
from .schemas import DatasetProfile


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
ARTIFACT_DIR = DATA_DIR / "artifacts"


@dataclass
class DatasetRecord:
    dataset_id: str
    original_name: str
    stored_path: Path
    loaded_name: str
    uploaded_at: datetime
    df: pd.DataFrame
    profile: DatasetProfile


@dataclass
class SessionRecord:
    messages: list[dict[str, Any]] = field(default_factory=list)
    previous_results: list[dict[str, Any]] = field(default_factory=list)


class AppState:
    def __init__(self) -> None:
        self.datasets: dict[str, DatasetRecord] = {}
        self.sessions: dict[str, SessionRecord] = {}

    def save_upload(self, filename: str, content: bytes) -> DatasetRecord:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dataset_id = uuid4().hex
        safe_name = Path(filename).name
        stored_path = UPLOAD_DIR / f"{dataset_id}_{safe_name}"
        stored_path.write_bytes(content)
        df, loaded_name = load_dataframe(stored_path)
        uploaded_at = datetime.now(timezone.utc)
        profile = profile_dataframe(dataset_id, loaded_name, df, uploaded_at)
        record = DatasetRecord(dataset_id, safe_name, stored_path, loaded_name, uploaded_at, df, profile)
        self.datasets[dataset_id] = record
        return record

    def get_dataset(self, dataset_id: str) -> DatasetRecord:
        if dataset_id not in self.datasets:
            raise KeyError(f"Dataset not found: {dataset_id}")
        return self.datasets[dataset_id]

    def get_session(self, session_id: str) -> SessionRecord:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionRecord()
        return self.sessions[session_id]


state = AppState()
