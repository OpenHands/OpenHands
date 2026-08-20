import { I18nKey } from "#/i18n/declaration";

/**
 * Structured `code` values the SDK's ACPAgent puts on a ConversationErrorEvent
 * (see software-agent-sdk `acp_agent.py`). The banner uses them to show a
 * code-specific header and, for credential failures, a recovery action.
 */
export const ACP_AUTH_REQUIRED_CODE = "ACPAuthRequired";

const ACP_ERROR_HEADER_KEYS: Record<string, I18nKey> = {
  ACPAuthRequired: I18nKey.ERROR$ACP_AUTH_REQUIRED_TITLE,
  // Spawn/init/prompt/usage-policy failures share the generic "Agent error"
  // header; their detail already carries the specific cause.
  ACPSpawnError: I18nKey.CHAT_INTERFACE$AGENT_ERROR_MESSAGE,
  ACPInitError: I18nKey.CHAT_INTERFACE$AGENT_ERROR_MESSAGE,
  ACPPromptError: I18nKey.CHAT_INTERFACE$AGENT_ERROR_MESSAGE,
  UsagePolicyRefusal: I18nKey.CHAT_INTERFACE$AGENT_ERROR_MESSAGE,
};

/** Localized header key for an error code, or null when the code is unknown. */
export function getAcpErrorHeaderKey(code?: string | null): I18nKey | null {
  if (!code) return null;
  return ACP_ERROR_HEADER_KEYS[code] ?? null;
}

/** Whether the error is a credential failure that warrants a re-auth action. */
export function isAcpAuthErrorCode(code?: string | null): boolean {
  return code === ACP_AUTH_REQUIRED_CODE;
}
