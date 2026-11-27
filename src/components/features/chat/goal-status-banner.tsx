import { useGoalStore } from "#/stores/goal-store";
import { GoalStatusContent } from "./goal-status-content";

export interface GoalStatusBannerProps {
  conversationId: string | null | undefined;
}

/**
 * Live, bottom-pinned banner for an *active* `/goal` loop, fed by goal
 * ConversationStateUpdateEvents (see useGoalStore). Once the loop ends it hides:
 * the final status is rendered inline in the message timeline instead (see
 * event-message.tsx + should-render-event.ts), so the finished goal settles into
 * the conversation and later messages append below it.
 */
export function GoalStatusBanner({ conversationId }: GoalStatusBannerProps) {
  const statusByConversation = useGoalStore((s) => s.statusByConversation);
  const status = conversationId
    ? statusByConversation[conversationId]
    : undefined;

  if (!status?.active) return null;

  return <GoalStatusContent status={status} />;
}
