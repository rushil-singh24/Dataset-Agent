from __future__ import annotations

import shutil
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
MAX_DATASETS_IN_MEMORY = 20


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

    def reset_runtime(self, *, clear_files: bool = False) -> None:
        """Clear all runtime data.

        This app is intended to be generic and local-first. Nothing from a previous
        local run should affect the next person/session after the server restarts.
        """
        self.datasets.clear()
        self.sessions.clear()

        if clear_files:
            for directory in (UPLOAD_DIR, ARTIFACT_DIR):
                if directory.exists():
                    shutil.rmtree(directory, ignore_errors=True)
                directory.mkdir(parents=True, exist_ok=True)

    def save_upload(self, filename: str, content: bytes) -> DatasetRecord:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dataset_id = uuid4().hex
        safe_name = Path(filename).name
        stored_path = UPLOAD_DIR / f"{dataset_id}_{safe_name}"
        stored_path.write_bytes(content)

        try:
            df, loaded_name = load_dataframe(stored_path)
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise

        uploaded_at = datetime.now(timezone.utc)
        profile = profile_dataframe(dataset_id, loaded_name, df, uploaded_at)
        record = DatasetRecord(dataset_id, safe_name, stored_path, loaded_name, uploaded_at, df, profile)
        self.datasets[dataset_id] = record
        self._evict_oldest_if_needed()
        return record

    def get_dataset(self, dataset_id: str) -> DatasetRecord:
        if dataset_id not in self.datasets:
            raise KeyError(f"Dataset not found: {dataset_id}")
        return self.datasets[dataset_id]

    def get_session(self, session_id: str) -> SessionRecord:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionRecord()
        return self.sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def _evict_oldest_if_needed(self) -> None:
        while len(self.datasets) > MAX_DATASETS_IN_MEMORY:
            oldest_id = min(self.datasets, key=lambda item: self.datasets[item].uploaded_at)
            record = self.datasets.pop(oldest_id)
            try:
                record.stored_path.unlink(missing_ok=True)
            except OSError:
                pass


state = AppState()
