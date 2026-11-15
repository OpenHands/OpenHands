import { ConversationClient } from "@openhands/typescript-client/clients";
import type { AskUserAnswerEvent } from "@openhands/typescript-client";
import { getAgentServerClientOptions } from "../agent-server-client-options";

/** Answers to the agent's `ask_user` questions (decision cards). */
export const askUserService = {
  /**
   * Picks an option of the pending question, with the user's own words when
   * they added any. The server records the answer, passes it to the agent as
   * the user's next message, and runs the agent.
   */
  answer: async ({
    conversationId,
    askId,
    optionIndex,
    text,
  }: {
    conversationId: string;
    askId: string;
    optionIndex?: number;
    text?: string;
  }): Promise<AskUserAnswerEvent> =>
    new ConversationClient(getAgentServerClientOptions()).answerAskUser(
      conversationId,
      askId,
      { optionIndex, text },
    ),
};
