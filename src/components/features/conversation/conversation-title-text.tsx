import { TextShimmer } from "#/components/shared/text-shimmer";

export function ConversationTitleText({
  title,
  pending = false,
}: {
  title: string;
  pending?: boolean;
}) {
  return pending ? (
    <TextShimmer as="span" aria-busy="true">
      {title}
    </TextShimmer>
  ) : (
    <span
      key={title}
      className="motion-safe:animate-[conversation-title-reveal_160ms_ease-out]"
    >
      {title}
    </span>
  );
}
