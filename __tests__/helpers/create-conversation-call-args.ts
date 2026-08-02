/** Trailing positional args for `AgentServerConversationService.createConversation`. */
export const CREATE_CONVERSATION_DEFAULT_TAIL = [
  undefined, // sandboxId
  undefined, // agentProfileId
  undefined, // agentProfileKind
  undefined, // agentSettingsOverride (Pi profile inlining)
] as const;
