import { LookupSecret, StaticSecret } from "@openhands/typescript-client";
import ConversationService from "./conversation-service.api";
import AgentServerConversationService from "./agent-server-conversation-service.api";
import { getEffectiveLocalBackend } from "../backend-registry/active-store";
import { buildAuthHeaders } from "../backend-registry/auth";

/**
 * Synchronize a newly added or updated secret to the active conversation's
 * SecretRegistry as a LookupSecret reference.
 *
 * When the active conversation executes its next bash command, the agent-server
 * resolves the secret lazily over HTTP without requiring a conversation restart.
 */
export async function syncSecretToActiveConversation(
  secretName: string,
  description?: string,
): Promise<void> {
  const currentConversation = ConversationService.getCurrentConversation();
  if (!currentConversation?.id) return;

  const backend = getEffectiveLocalBackend();
  const headers = backend ? buildAuthHeaders(backend) : {};

  const lookupSecret: LookupSecret = {
    kind: "LookupSecret",
    url: `/api/settings/secrets/${encodeURIComponent(secretName)}`,
    description,
  };

  if (Object.keys(headers).length > 0) {
    lookupSecret.headers = headers;
  }

  try {
    await AgentServerConversationService.updateSecrets(currentConversation.id, {
      [secretName]: lookupSecret,
    });
  } catch (error) {
    console.warn("Failed to sync secret to active conversation:", error);
  }
}

/**
 * Remove a deleted secret from the active conversation's SecretRegistry by
 * passing a null-valued StaticSecret placeholder.
 */
export async function removeSecretFromActiveConversation(
  secretName: string,
): Promise<void> {
  const currentConversation = ConversationService.getCurrentConversation();
  if (!currentConversation?.id) return;

  const staticSecret: StaticSecret = {
    kind: "StaticSecret",
    value: null,
  };

  try {
    await AgentServerConversationService.updateSecrets(currentConversation.id, {
      [secretName]: staticSecret,
    });
  } catch (error) {
    console.warn("Failed to remove secret from active conversation:", error);
  }
}

/**
 * Synchronize a secret rename to the active conversation: removes the old key
 * name and adds the new key name as a LookupSecret reference.
 */
export async function renameSecretInActiveConversation(
  oldName: string,
  newName: string,
  description?: string,
): Promise<void> {
  await removeSecretFromActiveConversation(oldName);
  await syncSecretToActiveConversation(newName, description);
}
