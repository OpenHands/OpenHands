export type PickerKind = "model" | "llm-profile";

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
  /** The current context runs an ACP agent (active conversation or active ACP profile). */
  isAcp: boolean;
}
// The chat-input pill is always an LLM selector (OSS-5735): the ACP model
// picker in an ACP context (constrained to that provider's models), the LLM
// profile picker otherwise — on every backend, before and during a
// conversation, so the pill always names the LLM profile rather than a raw
// model path. Whether a pick is actionable (cloud org permission, start-task
// route) is decided in ``useChatInputLlmProfileState``, not by swapping in a
// different pill. Agent-profile switching lives in the "+" tools menu while
// the conversation hasn't started, not here.
export function resolvePickerKind({
  isAcp,
}: ResolvePickerKindInput): PickerKind {
  return isAcp ? "model" : "llm-profile";
}
