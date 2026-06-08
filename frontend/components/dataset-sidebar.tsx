"use client";

import { Database, FileSpreadsheet } from "lucide-react";
import type { DatasetProfile } from "@/lib/api";
import { UploadZone } from "./upload-zone";

type DatasetSidebarProps = {
  dataset: DatasetProfile | null;
  onUpload: (file: File) => void;
  uploading: boolean;
  error: string | null;
};

export function DatasetSidebar({ dataset, onUpload, uploading, error }: DatasetSidebarProps) {
  return (
    <aside className="flex h-full w-full flex-col border-r border-border bg-panel lg:w-[360px]">
      <div className="border-b border-border p-5">
        <div className="flex items-center gap-2">
          <Database className="h-5 w-5 text-accent" aria-hidden />
          <h1 className="text-lg font-semibold">DataChat AI</h1>
        </div>
        <p className="mt-1 text-sm text-muted">Local analysis grounded in your uploaded dataset.</p>
      </div>

      <div className="space-y-5 overflow-y-auto p-5 scrollbar-thin">
        <UploadZone onUpload={onUpload} disabled={uploading} />
        {uploading ? <p className="text-sm text-muted">Profiling dataset...</p> : null}
        {error ? <p className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-danger">{error}</p> : null}

        {dataset ? (
          <div className="space-y-5">
            <section className="rounded-md border border-border p-4">
              <div className="flex items-start gap-3">
                <FileSpreadsheet className="mt-1 h-5 w-5 text-accent" aria-hidden />
                <div className="min-w-0">
                  <h2 className="truncate text-sm font-semibold">{dataset.metadata.name}</h2>
                  <p className="text-xs text-muted">{new Date(dataset.metadata.uploaded_at).toLocaleString()}</p>
                </div>
              </div>
              <div className="mt-4 grid grid-cols-3 gap-2 text-center">
                <Metric label="Rows" value={dataset.metadata.row_count.toLocaleString()} />
                <Metric label="Columns" value={dataset.metadata.column_count.toLocaleString()} />
                <Metric label="MB" value={dataset.metadata.memory_usage_mb.toLocaleString()} />
              </div>
            </section>

            <section>
              <h2 className="mb-2 text-sm font-semibold">Dataset Summary</h2>
              <pre className="whitespace-pre-wrap rounded-md border border-border bg-stone-50 p-3 text-xs leading-5 text-slate-700">
                {dataset.summary}
              </pre>
            </section>

            <section>
              <h2 className="mb-2 text-sm font-semibold">Columns</h2>
              <div className="space-y-2">
                {dataset.columns.map((column) => (
                  <div key={column.name} className="rounded-md border border-border px-3 py-2">
                    <div className="flex items-center justify-between gap-3">
                      <span className="truncate text-sm font-medium">{column.name}</span>
                      <span className="shrink-0 rounded bg-slate-100 px-2 py-1 text-[11px] text-muted">{column.dtype}</span>
                    </div>
                    <p className="mt-1 text-xs text-muted">
                      {column.missing_percent}% missing, {column.unique_count.toLocaleString()} unique
                    </p>
                  </div>
                ))}
              </div>
            </section>
          </div>
        ) : (
          <div className="rounded-md border border-border bg-stone-50 p-4 text-sm text-muted">
            Upload a dataset to populate schema, profiling, and chat context.
          </div>
        )}
      </div>
    </aside>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-stone-50 p-2">
      <div className="text-sm font-semibold">{value}</div>
      <div className="text-[11px] uppercase tracking-wide text-muted">{label}</div>
    </div>
  );
}
