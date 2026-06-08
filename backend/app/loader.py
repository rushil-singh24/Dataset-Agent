from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".tab",
    ".txt",
    ".json",
    ".jsonl",
    ".ndjson",
    ".xlsx",
    ".xls",
    ".parquet",
    ".feather",
    ".orc",
}


def load_dataframe(path: Path) -> tuple[pd.DataFrame, str]:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        return _load_zip(path)
    return _load_supported_file(path)


def _load_supported_file(path: Path) -> tuple[pd.DataFrame, str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path), path.name
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t"), path.name
    if suffix == ".txt":
        return pd.read_csv(path, sep=None, engine="python"), path.name
    if suffix == ".json":
        return _read_json(path), path.name
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True), path.name
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path), path.name
    if suffix == ".parquet":
        return pd.read_parquet(path), path.name
    if suffix == ".feather":
        return pd.read_feather(path), path.name
    if suffix == ".orc":
        return pd.read_orc(path), path.name
    raise ValueError(f"Unsupported file type: {suffix}")


def _read_json(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_json(path)
        if isinstance(frame, pd.DataFrame):
            return frame
    except ValueError:
        pass

    import json

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return pd.json_normalize(payload)
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                return pd.json_normalize(value)
        return pd.json_normalize(payload)
    raise ValueError("JSON file must contain an object, an array of objects, or newline-delimited objects.")


def _load_zip(path: Path) -> tuple[pd.DataFrame, str]:
    extract_dir = path.with_suffix("")
    extract_dir.mkdir(parents=True, exist_ok=True)
    extract_root = extract_dir.resolve()

    with ZipFile(path) as archive:
        candidates = [
            member
            for member in archive.namelist()
            if not member.endswith("/") and Path(member).suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not candidates:
            raise ValueError("ZIP does not contain a supported tabular file.")
        if len(candidates) > 1:
            names = ", ".join(candidates[:8])
            raise ValueError(f"ZIP contains multiple supported files. Upload one file at a time for MVP: {names}")

        selected_member = candidates[0]
        target_path = (extract_dir / selected_member).resolve()
        if not target_path.is_relative_to(extract_root):
            raise ValueError("ZIP contains an unsafe file path.")

        archive.extract(selected_member, extract_dir)

    return _load_supported_file(target_path)
