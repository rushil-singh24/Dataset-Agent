# DataChat AI

Local conversational data analysis for CSV, Excel, Parquet, and ZIP datasets.

DataChat AI runs on localhost with a Next.js frontend, FastAPI backend, Pandas/DuckDB analysis layer, and Ollama/Qwen3 for local agent reasoning. Every analytical answer is grounded in uploaded data, generated statistics, executed code, or visualizations derived from the dataset.

## Features

- Upload CSV, TSV/TAB/TXT, JSON/JSONL/NDJSON, XLSX, XLS, Parquet, Feather, ORC, or ZIP datasets.
- Generate immediate dataset profiling: schema, missing values, numerical stats, categorical values, duplicates, constant columns, and high-cardinality columns.
- Chat with the active dataset through a ChatGPT-style interface.
- Show a transparent analysis plan before every analytical response.
- Execute generated Pandas/DuckDB code with AST validation and restricted globals.
- Render result tables, generated charts, confidence indicators, execution status, and hidden-by-default generated code.
- Refuse unsupported questions that cannot be answered from the uploaded dataset.

## Dataset Grounding

The backend loads the full uploaded dataset into a dataframe and generated Pandas/DuckDB code runs against all rows and columns. For small and medium datasets, a full JSON-record snapshot is also included in the local model prompt. For larger datasets, the prompt includes schema, profiling, examples, and data-quality context while execution still runs over the complete dataframe.

## Local Setup

### From the repository root

```bash
npm run backend
npm run dev
```

### Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend setup

```bash
cd frontend
npm install
npm run dev
```

If you prefer to install frontend dependencies from the repo root:

```bash
npm --prefix frontend install
```

### Ollama

```bash
ollama pull qwen3:8b
ollama run qwen3:8b
```

Open `http://localhost:3000`.

## Configuration

Backend environment variables:

- `OLLAMA_BASE_URL`, default `http://localhost:11434`
- `OLLAMA_MODEL`, default `qwen3:8b`

Frontend environment variables:

- `NEXT_PUBLIC_API_BASE_URL`, default `http://127.0.0.1:8000`

## Notes

This MVP supports one active dataset per session. Multi-dataset comparison, predictive modeling, dashboards, scheduled reports, and natural-language SQL mode are planned Phase 2 capabilities.
