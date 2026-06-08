export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

export type ColumnProfile = {
  name: string;
  dtype: string;
  missing_count: number;
  missing_percent: number;
  unique_count: number;
  sample_values: unknown[];
  stats: Record<string, unknown>;
  top_values: Array<Record<string, unknown>>;
};

export type DatasetProfile = {
  metadata: {
    dataset_id: string;
    name: string;
    uploaded_at: string;
    row_count: number;
    column_count: number;
    columns: string[];
    dtypes: Record<string, string>;
    memory_usage_mb: number;
  };
  columns: ColumnProfile[];
  duplicate_rows: number;
  empty_columns: string[];
  constant_columns: string[];
  high_cardinality_columns: string[];
  summary: string;
};

export type ChatResponse = {
  assistant_message: string;
  classification: string;
  analysis_plan: {
    category: string;
    question_understanding: string;
    selected_columns: string[];
    column_selection_reason: string;
    planned_operations: string[];
    output_type: string;
    estimated_complexity: string;
    grounding_status: string;
    status: string;
  };
  execution: {
    status_steps: Array<{ label: string; state: string }>;
    generated_code?: string | null;
    tables: Array<{ title: string; columns: string[]; rows: Array<Record<string, unknown>> }>;
    charts: Array<{ artifact_id: string; title: string; type: string; url: string }>;
    raw_result: unknown;
  };
  confidence: {
    level: "High" | "Medium" | "Low";
    reason: string;
  };
};

export async function uploadDataset(file: File): Promise<DatasetProfile> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE_URL}/api/datasets/upload`, {
    method: "POST",
    body: form
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Upload failed");
  }
  return response.json();
}

export async function sendChat(datasetId: string, message: string, sessionId: string, educationalMode: boolean) {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId, message, session_id: sessionId, educational_mode: educationalMode })
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Chat request failed");
  }
  return response.json() as Promise<ChatResponse>;
}
