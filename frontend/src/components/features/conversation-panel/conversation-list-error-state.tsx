interface ConversationListErrorStateProps {
  onRetry: () => void;
}

export function ConversationListErrorState({
  onRetry,
}: ConversationListErrorStateProps) {
  return (
    <div
      data-testid="conversation-list-error-state"
      className="flex flex-col items-center justify-center gap-3 h-full px-4 py-6"
    >
      <p className="text-sm text-neutral-400 text-center">
        Failed to load conversations. Please try again.
      </p>
      <button
        type="button"
        onClick={onRetry}
        className="text-xs text-white underline hover:text-neutral-300"
      >
        Retry
      </button>
    </div>
  );
}
