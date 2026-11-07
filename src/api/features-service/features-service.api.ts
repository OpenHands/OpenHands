import {
  ConversationClient,
  SettingsClient,
  type FeatureFlag,
} from "@openhands/typescript-client/clients";
import type { ReactionEmoji } from "@openhands/typescript-client";
import { getActiveBackend } from "../backend-registry/active-store";
import { getAgentServerClientOptions } from "../agent-server-client-options";

/**
 * User-facing feature flags (`/api/settings/features`): yes/no switches the
 * user sets for themselves, read by the agent server when a conversation
 * starts. Local agent server only; Cloud has no such settings yet, so it lists
 * none.
 */
export const featuresService = {
  listFeatures: async (): Promise<FeatureFlag[]> => {
    if (getActiveBackend().backend.kind === "cloud") return [];
    return new SettingsClient(getAgentServerClientOptions()).listFeatures();
  },

  updateFeature: async ({
    key,
    enabled,
  }: {
    key: string;
    enabled: boolean;
  }): Promise<FeatureFlag> =>
    new SettingsClient(getAgentServerClientOptions()).updateFeature(
      key,
      enabled,
    ),

  /** Put the person's reaction on a message, or take it off. */
  react: async ({
    conversationId,
    emoji,
    targetEventId,
    removed,
  }: {
    conversationId: string;
    emoji: ReactionEmoji;
    targetEventId: string;
    removed: boolean;
  }): Promise<void> => {
    await new ConversationClient(getAgentServerClientOptions()).react(
      conversationId,
      emoji,
      { targetEventId, removed },
    );
  },
};
