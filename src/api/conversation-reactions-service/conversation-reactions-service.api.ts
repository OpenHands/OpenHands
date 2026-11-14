import { ConversationClient } from "@openhands/typescript-client/clients";
import type { ReactionEmoji } from "@openhands/typescript-client";
import { getAgentServerClientOptions } from "../agent-server-client-options";

/** Conversation behavior belongs here, not in the experimental flag registry. */
export const conversationReactionsService = {
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
