import { Pre } from "#/ui/pre";

interface SystemMessageContentProps {
  content: string;
}

export function SystemMessageContent({ content }: SystemMessageContentProps) {
  return (
    <Pre
      size="small"
      font="mono"
      lineHeight="relaxed"
      padding="medium"
      className="text-[var(--oh-text-tertiary)]"
    >
      {content}
    </Pre>
  );
}
