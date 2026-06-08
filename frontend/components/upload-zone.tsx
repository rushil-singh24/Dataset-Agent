"use client";

import { UploadCloud } from "lucide-react";
import { useRef, useState } from "react";
import { cn } from "@/lib/utils";

type UploadZoneProps = {
  onUpload: (file: File) => void;
  disabled?: boolean;
};

export function UploadZone({ onUpload, disabled }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  function handleFiles(files: FileList | null) {
    const file = files?.[0];
    if (file) onUpload(file);
  }

  return (
    <div
      className={cn(
        "rounded-md border border-dashed border-border bg-panel p-4 transition",
        isDragging && "border-accent bg-teal-50",
        disabled && "opacity-60"
      )}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setIsDragging(false);
        if (!disabled) handleFiles(event.dataTransfer.files);
      }}
    >
      <input
        ref={inputRef}
        className="hidden"
        type="file"
        accept=".csv,.tsv,.tab,.txt,.json,.jsonl,.ndjson,.xlsx,.xls,.parquet,.feather,.orc,.zip"
        onChange={(event) => handleFiles(event.target.files)}
        disabled={disabled}
      />
      <button
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        className="flex w-full items-center gap-3 rounded-md border border-border px-3 py-3 text-left hover:border-accent disabled:cursor-not-allowed"
      >
        <UploadCloud className="h-5 w-5 text-accent" aria-hidden />
        <span>
          <span className="block text-sm font-semibold">Upload dataset</span>
          <span className="block text-xs text-muted">CSV, JSON, Excel, Parquet, TSV, Feather, ORC, or ZIP</span>
        </span>
      </button>
    </div>
  );
}
