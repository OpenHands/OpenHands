export function TypingIndicator() {
  return (
    <div className="flex items-center space-x-1.5 rounded-full bg-[var(--oh-surface)] px-3 py-1.5">
      <span className="h-1.5 w-1.5 rounded-full bg-[var(--oh-muted)] animate-[bounce_0.5s_infinite] translate-y-[1px] [animation-delay:0ms]" />
      <span className="h-1.5 w-1.5 rounded-full bg-[var(--oh-muted)] animate-[bounce_0.5s_infinite] translate-y-[1px] [animation-delay:75ms]" />
      <span className="h-1.5 w-1.5 rounded-full bg-[var(--oh-muted)] animate-[bounce_0.5s_infinite] translate-y-[1px] [animation-delay:150ms]" />
    </div>
  );
}
