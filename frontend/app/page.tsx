"use client";

import { Bot, SendHorizontal, Settings2 } from "lucide-react";
import { FormEvent, useState } from "react";
import { DatasetSidebar } from "@/components/dataset-sidebar";
import { ChatMessage, MessageCard } from "@/components/message-card";
import { sendChat, uploadDataset, type DatasetProfile } from "@/lib/api";

export default function Home() {
  const [dataset, setDataset] = useState<DatasetProfile | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [uploading, setUploading] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [educationalMode, setEducationalMode] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID());

  async function handleUpload(file: File) {
    setUploading(true);
    setError(null);

    try {
      const profile = await uploadDataset(file);
      const newSessionId = crypto.randomUUID();

      setSessionId(newSessionId);
      setDataset(profile);
      setInput("");

      setMessages([
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `Dataset loaded: ${profile.metadata.name}. I profiled ${profile.metadata.row_count.toLocaleString()} rows and ${profile.metadata.column_count.toLocaleString()} columns. Ask a question grounded in this data.`,
          response: {
            assistant_message: `Dataset loaded: ${profile.metadata.name}. I profiled ${profile.metadata.row_count.toLocaleString()} rows and ${profile.metadata.column_count.toLocaleString()} columns. Ask a question grounded in this data.`,
            classification: "Dataset Information",
            analysis_plan: {
              category: "Dataset Information",
              question_understanding: "Dataset upload and profiling",
              selected_columns: [],
              column_selection_reason:
                "All columns were inspected during profiling.",
              planned_operations: [
                "Load file",
                "Infer schema",
                "Profile missing values and summary statistics",
              ],
              output_type: "Text",
              estimated_complexity: "Low",
              grounding_status: "Ready",
              status: "Complete",
            },
            execution: {
              status_steps: [
                { label: "Load Dataset", state: "complete" },
                { label: "Profile Dataset", state: "complete" },
                { label: "Complete", state: "complete" },
              ],
              generated_code: null,
              tables: [],
              charts: [],
              raw_result: {
                dataset_name: profile.metadata.name,
                row_count: profile.metadata.row_count,
                column_count: profile.metadata.column_count,
              },
            },
            confidence: {
              level: "High",
              reason:
                "Profile was generated directly from the uploaded dataset.",
            },
          },
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const content = input.trim();

    if (!content || !dataset || thinking) return;

    setInput("");
    setThinking(true);
    setError(null);

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content,
    };

    setMessages((current) => [...current, userMessage]);

    try {
      const response = await sendChat(
        dataset.metadata.dataset_id,
        content,
        sessionId,
        educationalMode
      );

      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: response.assistant_message,
          response,
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Chat request failed");
    } finally {
      setThinking(false);
    }
  }

  return (
    <main className="flex h-screen overflow-hidden">
      <DatasetSidebar
        dataset={dataset}
        onUpload={handleUpload}
        uploading={uploading}
        error={error}
        readOnly={false}
      />

      <section className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-border bg-panel px-5 py-3">
          <div className="flex items-center gap-2">
            <Bot className="h-5 w-5 text-accent" aria-hidden />

            <div>
              <h2 className="text-sm font-semibold">Dataset Chat</h2>

              <p className="text-xs text-muted">
                {dataset
                  ? "Grounded in uploaded data"
                  : "Upload a dataset to begin"}
              </p>
            </div>
          </div>

          <label className="flex items-center gap-2 rounded-md border border-border px-3 py-2 text-sm">
            <Settings2 className="h-4 w-4 text-accent" aria-hidden />
            <span>Explain the Analysis</span>

            <input
              type="checkbox"
              checked={educationalMode}
              onChange={(event) =>
                setEducationalMode(event.target.checked)
              }
              className="h-4 w-4 accent-teal-700"
            />
          </label>
        </header>

        <div className="flex-1 overflow-y-auto p-5 scrollbar-thin">
          <div className="mx-auto max-w-5xl space-y-5">
            {messages.length === 0 ? (
              <div className="rounded-md border border-border bg-panel p-6 shadow-soft">
                <h2 className="text-lg font-semibold">
                  Upload a dataset and start asking questions.
                </h2>

                <p className="mt-2 text-sm leading-6 text-muted">
                  Try asking about missing values, correlations, top
                  categories, trends over time, or charts once profiling is
                  complete.
                </p>
              </div>
            ) : null}

            {messages.map((message) => (
              <MessageCard
                key={message.id}
                message={message}
                showAnalysis={educationalMode}
              />
            ))}

            {thinking ? (
              <div className="rounded-md border border-border bg-panel p-4 text-sm text-muted shadow-soft">
                Analyzing the uploaded dataset...
              </div>
            ) : null}
          </div>
        </div>

        <form
          onSubmit={handleSubmit}
          className="border-t border-border bg-panel p-4"
        >
          <div className="mx-auto flex max-w-5xl gap-3">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              disabled={!dataset || thinking}
              rows={1}
              placeholder={
                dataset
                  ? "Ask a question about this dataset..."
                  : "Upload a dataset first"
              }
              className="min-h-11 flex-1 resize-none rounded-md border border-border px-4 py-3 text-sm outline-none focus:border-accent disabled:bg-stone-100"
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
            />

            <button
              type="submit"
              disabled={!dataset || thinking || !input.trim()}
              className="inline-flex h-11 w-11 items-center justify-center rounded-md bg-accent text-white hover:opacity-90 disabled:cursor-not-allowed disabled:bg-slate-300"
              title="Send"
            >
              <SendHorizontal className="h-5 w-5" aria-hidden />
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}