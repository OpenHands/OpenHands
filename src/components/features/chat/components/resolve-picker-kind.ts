export type PickerKind = "model" | "agent-profile" | "llm-profile";

export interface ConversationStartState {
  isLoadingHistory: boolean;
  hasUserEvents: boolean;
  hasPendingUserMessages: boolean;
  hasSubstantiveAgentActions: boolean;
  hasModelEntries: boolean;
}

export function hasConversationStarted({
  isLoadingHistory,
  hasUserEvents,
  hasPendingUserMessages,
  hasSubstantiveAgentActions,
  hasModelEntries,
}: ConversationStartState) {
  return (
    isLoadingHistory ||
    hasUserEvents ||
    hasPendingUserMessages ||
    hasSubstantiveAgentActions ||
    hasModelEntries
  );
}

export interface ResolvePickerKindInput {
  hasConversation: boolean;
  hasStartedConversation?: boolean;
  isCloud: boolean;
  isAcp: boolean;
  profilesAvailable: boolean;
}
export function resolvePickerKind({
  hasConversation,
  hasStartedConversation = hasConversation,
  isCloud,
  isAcp,
  profilesAvailable,
}: ResolvePickerKindInput): PickerKind {
  if (!hasConversation || !hasStartedConversation) {
    if (profilesAvailable) return "agent-profile";
    return isCloud ? "model" : "llm-profile";
  }
  return isAcp ? "model" : "llm-profile";
}
