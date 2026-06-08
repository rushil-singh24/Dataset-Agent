"use client";

type ConfirmationModalProps = {
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
};

export function ConfirmationModal({
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  onConfirm,
  onCancel
}: ConfirmationModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4">
      <div className="w-full max-w-md rounded-md border border-border bg-panel p-5 shadow-soft">
        <h2 className="text-base font-semibold">{title}</h2>
        <p className="mt-2 text-sm leading-6 text-muted">{message}</p>
        <div className="mt-5 flex justify-end gap-3">
          <button type="button" onClick={onCancel} className="rounded-md border border-border px-4 py-2 text-sm hover:border-accent">
            {cancelLabel}
          </button>
          <button type="button" onClick={onConfirm} className="rounded-md bg-accent px-4 py-2 text-sm text-white hover:opacity-90">
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
