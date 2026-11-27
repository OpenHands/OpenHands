import { useLayoutEffect } from "react";
import { ConversationLoading } from "../../conversation-loading";

export function SuspensePendingFallback({
  onPending,
}: {
  onPending: () => void;
}) {
  useLayoutEffect(() => {
    onPending();
  }, [onPending]);

  return <ConversationLoading />;
}
