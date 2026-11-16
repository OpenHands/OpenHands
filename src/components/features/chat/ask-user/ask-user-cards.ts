import type { AskUserAnswerEvent } from "@openhands/typescript-client";
import type { MessageEvent, OpenHandsEvent } from "#/types/agent-server/core";
import {
  isActionEvent,
  isMessageEvent,
  isObservationEvent,
} from "#/types/agent-server/type-guards";

/** The experimental feature that gives the agent `ask_user` (SDK registry). */
export const ASK_USER_EXPERIMENTAL_FEATURE = "ask_user";

const ASK_USER_TOOL_NAME = "ask_user";

/**
 * Opening of the user message the server builds from an answer picked on the
 * card (`ANSWER_HEADER_PREFIX` in openhands.sdk.event.ask_user).
 */
export const ASK_USER_ANSWER_PREFIX = "[Answer to your question";

export interface AskUserOption {
  label: string;
  consequence: string;
}

export type AskUserStatus =
  /** Waiting for the user: a tap or a chat reply answers it. */
  | { kind: "pending" }
  /** Picked on the card. */
  | {
      kind: "answered";
      optionIndex: number | null;
      optionLabel: string | null;
      text: string | null;
    }
  /** The user wrote in the chat instead. */
  | { kind: "answered_in_chat" }
  /** The agent asked again before the user answered; only the latest waits. */
  | { kind: "closed" };

export interface AskUserCard {
  /** The `ask_user` call's ActionEvent id, which an answer names. */
  askId: string;
  question: string;
  options: AskUserOption[];
  recommended: number | null;
  status: AskUserStatus;
}

export interface AskUserCards {
  /** Cards by the id of the `ask_user` observation they replace in the chat. */
  byObservationId: ReadonlyMap<string, AskUserCard>;
  /** User messages that carry a card answer; the card shows them instead. */
  answerMessageIds: ReadonlySet<string>;
}

const EMPTY: AskUserCards = {
  byObservationId: new Map(),
  answerMessageIds: new Set(),
};

const isAskUserAnswerEvent = (
  event: OpenHandsEvent,
): event is AskUserAnswerEvent =>
  "kind" in event && event.kind === "AskUserAnswerEvent";

/**
 * The user's own message: relayed messages (another conversation, a `/goal`
 * loop) carry a `sender` and never answer a question.
 */
const isUserOwnMessage = (event: OpenHandsEvent): event is MessageEvent =>
  isMessageEvent(event) &&
  event.source === "user" &&
  !(event as { sender?: string | null }).sender;

const messageText = (event: MessageEvent): string => {
  const { content } = event.llm_message;
  return Array.isArray(content)
    ? content
        .filter((c) => c.type === "text")
        .map((c) => c.text)
        .join("\n")
    : "";
};

const readQuestion = (
  action: unknown,
): Omit<AskUserCard, "askId" | "status"> | null => {
  if (!action || typeof action !== "object") return null;
  const { question, options, recommended } = action as Record<string, unknown>;
  if (typeof question !== "string" || !Array.isArray(options)) return null;
  return {
    question,
    options: options.map((option) => {
      const { label, consequence } = (option ?? {}) as Record<string, unknown>;
      return {
        label: typeof label === "string" ? label : "",
        consequence: typeof consequence === "string" ? consequence : "",
      };
    }),
    recommended: typeof recommended === "number" ? recommended : null,
  };
};

/**
 * The decision cards in a conversation's events, oldest first, read the way
 * the server reads them (`find_pending_ask`): a successful `ask_user` call is
 * a question, and it waits until the user picks an option or writes in the
 * chat. Only the latest question can wait.
 */
export function buildAskUserCards(
  events: readonly OpenHandsEvent[],
): AskUserCards {
  let cards: Map<string, AskUserCard> | null = null;
  const byAskId = new Map<string, AskUserCard>();
  const answerMessageIds = new Set<string>();
  const questions = new Map<string, Omit<AskUserCard, "askId" | "status">>();
  let waiting: AskUserCard | null = null;
  // A tap saves its answer event, then the message the agent reads; only
  // that message closes the question (as on the server), so a tap whose
  // message never landed leaves the card waiting.
  let tapped: { card: AskUserCard; status: AskUserStatus } | null = null;

  for (const event of events) {
    if (isActionEvent(event) && event.tool_name === ASK_USER_TOOL_NAME) {
      const question = readQuestion(event.action);
      if (question) questions.set(event.id, question);
    } else if (
      isObservationEvent(event) &&
      event.tool_name === ASK_USER_TOOL_NAME &&
      !(event.observation as { is_error?: boolean }).is_error
    ) {
      const question = questions.get(event.action_id);
      if (question) {
        if (waiting) waiting.status = { kind: "closed" };
        waiting = {
          askId: event.action_id,
          ...question,
          status: { kind: "pending" },
        };
        cards ??= new Map();
        cards.set(event.id, waiting);
        byAskId.set(waiting.askId, waiting);
      }
    } else if (isAskUserAnswerEvent(event)) {
      const card = byAskId.get(event.ask_id);
      if (card && card === waiting) {
        tapped = {
          card,
          status: {
            kind: "answered",
            optionIndex: event.option_index ?? null,
            optionLabel: event.option_label ?? null,
            text: event.text ?? null,
          },
        };
      }
    } else if (isUserOwnMessage(event)) {
      if (
        tapped?.card === waiting &&
        tapped &&
        messageText(event).startsWith(ASK_USER_ANSWER_PREFIX)
      ) {
        tapped.card.status = tapped.status;
        answerMessageIds.add(event.id);
      } else if (waiting) {
        waiting.status = { kind: "answered_in_chat" };
      }
      waiting = null;
      tapped = null;
    }
  }

  return cards ? { byObservationId: cards, answerMessageIds } : EMPTY;
}

export const NO_ASK_USER_CARDS = EMPTY;
