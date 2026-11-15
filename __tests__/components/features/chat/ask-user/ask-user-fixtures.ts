import type { AskUserAnswerEvent } from "@openhands/typescript-client";
import type {
  ActionEvent,
  MessageEvent,
  ObservationEvent,
} from "#/types/agent-server/core";
import { SecurityRisk } from "#/types/agent-server/core";

const at = (n: number) => `2026-11-15T00:00:${String(n).padStart(2, "0")}Z`;

export const QUESTION = "Which database should config.yaml use?";

export const askAction = (id = "ask-1", n = 1): ActionEvent =>
  ({
    id,
    timestamp: at(n),
    source: "agent",
    thought: [{ type: "text", text: "This one is your call." }],
    thinking_blocks: [],
    action: {
      kind: "AskUserAction",
      question: QUESTION,
      options: [
        { label: "Postgres", consequence: "Needs a managed database." },
        { label: "SQLite", consequence: "One file, no server." },
      ],
      recommended: 0,
    },
    tool_name: "ask_user",
    tool_call_id: `call-${id}`,
    tool_call: {
      id: `call-${id}`,
      type: "function",
      function: { name: "ask_user", arguments: "{}" },
    },
    llm_response_id: `resp-${id}`,
    security_risk: SecurityRisk.UNKNOWN,
  }) as unknown as ActionEvent;

export const askShown = (
  actionId = "ask-1",
  n = 2,
  isError = false,
): ObservationEvent =>
  ({
    id: `obs-${actionId}`,
    timestamp: at(n),
    source: "environment",
    tool_name: "ask_user",
    tool_call_id: `call-${actionId}`,
    action_id: actionId,
    observation: {
      kind: "AskUserObservation",
      content: [{ type: "text", text: "The question is shown to the user." }],
      is_error: isError,
    },
  }) as unknown as ObservationEvent;

export const answer = (
  askId = "ask-1",
  n = 3,
  extra: Partial<AskUserAnswerEvent> = {},
): AskUserAnswerEvent => ({
  id: `answer-${askId}`,
  kind: "AskUserAnswerEvent",
  timestamp: at(n),
  source: "user",
  ask_id: askId,
  option_index: 1,
  option_label: "SQLite",
  text: null,
  ...extra,
});

export const userMessage = (
  id: string,
  text: string,
  n: number,
  sender: string | null = null,
): MessageEvent =>
  ({
    id,
    timestamp: at(n),
    source: "user",
    sender,
    llm_message: { role: "user", content: [{ type: "text", text }] },
    activated_skills: [],
    extended_content: [],
  }) as MessageEvent;

export const answerMessage = (id = "answer-msg", n = 4) =>
  userMessage(id, `[Answer to your question: "${QUESTION}"]\nSQLite`, n);
