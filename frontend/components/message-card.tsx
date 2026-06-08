"use client";

import { CheckCircle2, ChevronDown, Code2, Loader2, ShieldCheck, XCircle } from "lucide-react";
import Image from "next/image";
import { useState } from "react";
import type { ChatResponse } from "@/lib/api";
import { API_BASE_URL } from "@/lib/api";
import { cn } from "@/lib/utils";

export type ChatMessage =
  | { id: string; role: "user"; content: string }
  | { id: string; role: "assistant"; content: string; response: ChatResponse };

export function MessageCard({ message, showAnalysis }: { message: ChatMessage; showAnalysis: boolean }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[78%] rounded-md bg-slate-900 px-4 py-3 text-sm text-white">{message.content}</div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl space-y-3">
      {showAnalysis ? <PlanPanel response={message.response} /> : null}
      <Results response={message.response} showAnalysis={showAnalysis} />
      <div className="rounded-md border border-border bg-panel p-4 shadow-soft">
        <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
          <ShieldCheck className="h-4 w-4 text-accent" aria-hidden />
          Insights
        </div>
        <p className="whitespace-pre-wrap text-sm leading-6">{message.content}</p>
        <div className="mt-4 rounded-md bg-stone-50 p-3 text-xs text-muted">
          <span className="font-semibold text-foreground">Confidence: {message.response.confidence.level}.</span>{" "}
          {message.response.confidence.reason}
        </div>
      </div>
    </div>
  );
}

function PlanPanel({ response }: { response: ChatResponse }) {
  const [open, setOpen] = useState(true);
  return (
    <section className="rounded-md border border-border bg-panel shadow-soft">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left"
      >
        <span>
          <span className="block text-sm font-semibold">Analysis Plan</span>
          <span className="block text-xs text-muted">{response.classification}</span>
        </span>
        <ChevronDown className={cn("h-4 w-4 transition", open && "rotate-180")} aria-hidden />
      </button>
      {open ? (
        <div className="grid gap-4 border-t border-border p-4 lg:grid-cols-[1fr_260px]">
          <div className="space-y-3 text-sm">
            <PlanLine label="Question Understanding" value={response.analysis_plan.question_understanding} />
            <PlanLine label="Selected Columns" value={response.analysis_plan.selected_columns.join(", ") || "None"} />
            <PlanLine label="Reason" value={response.analysis_plan.column_selection_reason} />
            <div>
              <div className="text-xs font-semibold uppercase tracking-wide text-muted">Operations</div>
              <ul className="mt-1 list-disc space-y-1 pl-5">
                {response.analysis_plan.planned_operations.map((operation) => (
                  <li key={operation}>{operation}</li>
                ))}
              </ul>
            </div>
            <PlanLine label="Output Type" value={response.analysis_plan.output_type} />
            <PlanLine label="Grounding" value={`${response.analysis_plan.grounding_status}: ${response.analysis_plan.status}`} />
          </div>
          <StatusList steps={response.execution.status_steps} />
        </div>
      ) : null}
    </section>
  );
}

function PlanLine({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs font-semibold uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-1 text-sm">{value}</div>
    </div>
  );
}

function StatusList({ steps }: { steps: Array<{ label: string; state: string }> }) {
  return (
    <div className="rounded-md bg-stone-50 p-3">
      <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">Execution</div>
      <div className="space-y-2">
        {steps.map((step) => (
          <div key={`${step.label}-${step.state}`} className="flex items-center gap-2 text-xs">
            {step.state === "complete" ? <CheckCircle2 className="h-4 w-4 text-accent" /> : null}
            {step.state === "running" ? <Loader2 className="h-4 w-4 animate-spin text-slate-700" /> : null}
            {step.state === "blocked" ? <XCircle className="h-4 w-4 text-danger" /> : null}
            {step.state === "skipped" ? <span className="h-4 w-4 rounded-full border border-border" /> : null}
            <span>{step.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Results({ response, showAnalysis }: { response: ChatResponse; showAnalysis: boolean }) {
  const [showCode, setShowCode] = useState(false);
  const hasCode = Boolean(response.execution.generated_code);
  return (
    <div className="space-y-3">
      {response.execution.charts.map((chart) => (
        <div key={chart.artifact_id} className="rounded-md border border-border bg-panel p-4 shadow-soft">
          <div className="mb-3 text-sm font-semibold">{chart.title}</div>
          <Image
            className="h-auto w-full rounded-md border border-border"
            alt={chart.title}
            src={`${API_BASE_URL}${chart.url}`}
            width={1200}
            height={700}
            unoptimized
          />
        </div>
      ))}

      {showAnalysis
        ? response.execution.tables.map((table) => (
            <div key={table.title} className="overflow-hidden rounded-md border border-border bg-panel shadow-soft">
              <div className="border-b border-border px-4 py-3 text-sm font-semibold">{table.title}</div>
              <div className="max-h-[360px] overflow-auto">
                <table className="w-full min-w-[640px] border-collapse text-sm">
                  <thead className="sticky top-0 bg-stone-50">
                    <tr>
                      {table.columns.map((column) => (
                        <th key={column} className="border-b border-border px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-muted">
                          {column}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {table.rows.map((row, index) => (
                      <tr key={index} className="odd:bg-white even:bg-stone-50">
                        {table.columns.map((column) => (
                          <td key={column} className="border-b border-border px-3 py-2 align-top">
                            {String(row[column] ?? "")}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))
        : null}

      {showAnalysis && hasCode ? (
        <div className="rounded-md border border-border bg-panel">
          <button type="button" className="flex w-full items-center gap-2 px-4 py-3 text-sm font-semibold" onClick={() => setShowCode((value) => !value)}>
            <Code2 className="h-4 w-4 text-accent" aria-hidden />
            Show Generated Code
          </button>
          {showCode ? <pre className="overflow-auto border-t border-border bg-slate-950 p-4 text-xs leading-5 text-slate-100">{response.execution.generated_code}</pre> : null}
        </div>
      ) : null}
    </div>
  );
}
