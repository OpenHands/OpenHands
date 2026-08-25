import { cn } from "#/utils/utils";

interface HighlightSearchMatchProps {
  text: string;
  query: string;
  className?: string;
  highlightClassName?: string;
}

export function HighlightSearchMatch({
  text,
  query,
  className,
  highlightClassName = "text-white",
}: HighlightSearchMatchProps) {
  const trimmedQuery = query.trim();
  if (!trimmedQuery) {
    return <span className={className}>{text}</span>;
  }

  const lowerText = text.toLowerCase();
  const lowerQuery = trimmedQuery.toLowerCase();
  const matchIndex = lowerText.indexOf(lowerQuery);

  if (matchIndex === -1) {
    return <span className={className}>{text}</span>;
  }

  const before = text.slice(0, matchIndex);
  const match = text.slice(matchIndex, matchIndex + trimmedQuery.length);
  const after = text.slice(matchIndex + trimmedQuery.length);

  return (
    <span className={className}>
      {before}
      <span className={cn("font-semibold", highlightClassName)}>{match}</span>
      {after}
    </span>
  );
}
