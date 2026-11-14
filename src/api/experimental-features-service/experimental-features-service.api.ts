import {
  ConversationClient,
  SettingsClient,
  type ExperimentalFeature,
} from "@openhands/typescript-client/clients";
import type { ReactionEmoji } from "@openhands/typescript-client";
import type { Backend } from "../backend-registry/types";
import { getAgentServerClientOptions } from "../agent-server-client-options";

/**
 * Experimental features (`/api/settings/experimental`): yes/no switches the
 * user sets for themselves, read by the agent server when a conversation
 * starts. Local agent server only; Cloud has no such settings yet, so it lists
 * none.
 *
 * Every call names the backend it talks to. Each backend keeps its own
 * switches, and the user can change the active backend while a request is
 * queued or in flight, so the caller fixes the backend when the user acts
 * (see useExperimentalFeatures and useUpdateExperimentalFeature).
 */
const settingsClient = (backend: Backend) =>
  new SettingsClient(
    getAgentServerClientOptions({ host: backend.host, apiKey: backend.apiKey }),
  );

export const experimentalFeaturesService = {
  listExperimentalFeatures: async (
    backend: Backend,
  ): Promise<ExperimentalFeature[]> => {
    if (backend.kind === "cloud") return [];
    return settingsClient(backend).listExperimentalFeatures();
  },

  updateExperimentalFeature: async ({
    backend,
    key,
    enabled,
  }: {
    backend: Backend;
    key: string;
    enabled: boolean;
  }): Promise<ExperimentalFeature> =>
    settingsClient(backend).updateExperimentalFeature(key, enabled),

  /** Put the user's reaction on a message, or take it off. */
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
