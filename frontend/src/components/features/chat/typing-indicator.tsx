export function TypingIndicator() {
  return (
    <div className="flex items-center space-x-1.5 bg-tertiary px-3 py-1.5 rounded-full">
      <span
        className="w-1.5 h-1.5 bg-muted rounded-full animate-[bounce_0.5s_infinite] translate-y-px"
        style={{ animationDelay: "0ms" }}
      />
      <span
        className="w-1.5 h-1.5 bg-muted rounded-full animate-[bounce_0.5s_infinite] translate-y-px"
        style={{ animationDelay: "75ms" }}
      />
      <span
        className="w-1.5 h-1.5 bg-muted rounded-full animate-[bounce_0.5s_infinite] translate-y-px"
        style={{ animationDelay: "150ms" }}
      />
    </div>
  );
}
