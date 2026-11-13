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
    <span>{title}</span>
  );
}
