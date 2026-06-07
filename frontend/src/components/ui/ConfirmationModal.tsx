interface ConfirmationModalProps {
  isOpen: boolean;
  phrase: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmationModal({
  isOpen,
  phrase,
  onConfirm,
  onCancel
}: ConfirmationModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/5 px-4">
      <div className="bg-white border border-primary p-stack-lg max-w-md w-full rounded-xl animate-in fade-in zoom-in duration-300">
        <h2 className="text-headline-lg mb-stack-sm text-center">
          Send "{phrase}"?
        </h2>
        <div className="flex flex-col gap-4">
          <button
            onClick={onConfirm}
            className="w-full bg-primary text-white py-6 text-headline-md uppercase tracking-widest hover:opacity-90 transition-all active:scale-[0.98]"
          >
            Confirm
          </button>
          <button
            onClick={onCancel}
            className="w-full py-4 text-label-lg uppercase tracking-widest text-on-surface-variant border border-secondary-container rounded-lg hover:bg-surface-container-low transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
