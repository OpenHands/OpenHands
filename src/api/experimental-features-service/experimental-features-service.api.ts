import {
  ConversationClient,
  SettingsClient,
  type ExperimentalFeature,
} from "@openhands/typescript-client/clients";
import type { ReactionEmoji } from "@openhands/typescript-client";
import { getActiveBackend } from "../backend-registry/active-store";
import { getAgentServerClientOptions } from "../agent-server-client-options";

/**
 * Experimental features (`/api/settings/experimental`): yes/no switches the
 * user sets for themselves, read by the agent server when a conversation
 * starts. Local agent server only; Cloud has no such settings yet, so it lists
 * none.
 */
export const experimentalFeaturesService = {
  listExperimentalFeatures: async (): Promise<ExperimentalFeature[]> => {
    if (getActiveBackend().backend.kind === "cloud") return [];
    return new SettingsClient(
      getAgentServerClientOptions(),
    ).listExperimentalFeatures();
  },

  updateExperimentalFeature: async ({
    key,
    enabled,
  }: {
    key: string;
    enabled: boolean;
  }): Promise<ExperimentalFeature> =>
    new SettingsClient(getAgentServerClientOptions()).updateExperimentalFeature(
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
