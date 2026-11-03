/**
 * Superfast Decision Gate — System One front-door classifier (shadow mode).
 *
 * Concept and reference implementation by Andrea Bruno, released under Creative
 * Commons Attribution 4.0 (CC BY 4.0). See the harness-superfast white paper for
 * the full design. The decision models themselves (Von, OpenJev, Laya) are
 * third-party open models; only this integration architecture and the routing
 * method are covered by that attribution.
 *
 * A small, non-autoregressive "System One" decision server evaluates the pending
 * user turn in a single forward pass and returns typed, calibrated answers
 * without generating text. The gate turns those answers into a conservative
 * routing recommendation the harness could later act on.
 *
 * Design contract for this first increment:
 *   - Off by default. Nothing runs unless VITE_SUPERFAST_ENABLED is "true".
 *   - Shadow mode. The gate only classifies and logs. It never changes routing,
 *     never skips the model call, and never alters user-visible behavior.
 *   - Fail open. Any error, timeout, non-2xx response, or malformed body returns
 *     "no opinion" (null). It never throws into the caller and never adds latency
 *     to the real turn.
 *   - No new dependencies. It talks to a local Jev-compatible endpoint with the
 *     axios client the app already uses. The decision model is installed out of
 *     band, never bundled.
 */

import axios, { type AxiosRequestConfig } from "axios";
import EventLogger from "#/utils/event-logger";
import type {
  MessageContent,
  MessageTextContent,
  SendMessageRequest,
} from "#/api/conversation-service/agent-server-conversation-service.types";

/** Wire-level question kinds supported by the Jev-compatible protocol. */
type QuestionSpec =
  | { type: "noul"; instructions: string }
  | { type: "choice"; instructions: string; criteria: Record<string, string> };

/** A single answer returned by the decision model. */
interface DecisionAnswer {
  /** For `choice`: the selected criterion key. */
  choice?: string;
  /** For `noul`: probability the statement is true, in [0, 1]. */
  noul?: number;
  /** For `score`: expected value across the ordered levels. */
  score?: number;
  /** Calibrated confidence for choice / score answers. */
  confidence?: number;
  /** Full probability distribution for choice answers. */
  probabilities?: Record<string, number>;
}

/** Raw response envelope from the decision endpoint. */
interface SystemOneResponse {
  answers?: Record<string, DecisionAnswer>;
}

/** Runtime configuration for the gate, resolved from build-time env vars. */
export interface DecisionGateSettings {
  /** Master switch. When false the gate is never invoked. */
  enabled: boolean;
  /** Full URL of the decision endpoint, e.g. http://localhost:8000/v1/systemone */
  endpoint: string;
  /** Model id sent in the request body, e.g. von-1.2.0 */
  model: string;
  /** Hard timeout for a single decision call, in milliseconds. */
  timeoutMs: number;
}

const DEFAULT_ENDPOINT = "http://localhost:8000/v1/systemone";
const DEFAULT_MODEL = "von-1.2.0";
const DEFAULT_TIMEOUT_MS = 150;

/**
 * Read the gate settings from Vite build-time env vars. The gate is off unless
 * VITE_SUPERFAST_ENABLED is exactly the string "true".
 */
export function resolveGateSettings(): DecisionGateSettings {
  const rawTimeout = Number(import.meta.env.VITE_SUPERFAST_TIMEOUT_MS);
  const timeoutMs =
    Number.isInteger(rawTimeout) && rawTimeout > 0
      ? rawTimeout
      : DEFAULT_TIMEOUT_MS;
  return {
    enabled: import.meta.env.VITE_SUPERFAST_ENABLED === "true",
    endpoint: import.meta.env.VITE_SUPERFAST_ENDPOINT || DEFAULT_ENDPOINT,
    model: import.meta.env.VITE_SUPERFAST_MODEL || DEFAULT_MODEL,
    timeoutMs,
  };
}

/** Standard question set for classifying an incoming user turn. Kept small so a
 * single forward pass stays well under the timeout budget. */
const TURN_QUESTIONS: Record<string, QuestionSpec> = {
  needs_tool: {
    type: "noul",
    instructions:
      "Does answering this request require taking an action with a tool (reading, writing, running, searching), rather than replying from what is already known?",
  },
  answerable_from_context: {
    type: "noul",
    instructions:
      "Can this request be answered from information already present in the conversation, without any new investigation?",
  },
  intent: {
    type: "choice",
    instructions: "Classify the primary intent of the user request.",
    criteria: {
      code_change: "Create, edit, or delete code or files.",
      code_question: "Explain or reason about code without changing it.",
      command: "Run a command or operation.",
      chat: "Casual conversation or a question needing no tools.",
      other: "None of the above.",
    },
  },
};

/** The routing recommendation derived from a turn's decision answers. */
export type TurnRoute =
  | "needs_tool"
  | "answer_from_context"
  | "plain_chat"
  | "unknown";

/** A turn-level decision: the derived route plus measured latency. */
export interface TurnDecision {
  route: TurnRoute;
  latencyMs: number;
}

/** Minimum calibrated intent confidence required for the plain_chat route. */
const PLAIN_CHAT_CONFIDENCE_FLOOR = 0.5;

/**
 * Read a noul probability, returning it only when it is a real, finite value in
 * the closed [0, 1] interval. Anything else (absent, NaN, Infinity, out of range,
 * wrong type) is treated as "no evidence" (undefined), so a mis-scaled or
 * missing answer can never produce a decisive fast route.
 */
function readNoul(answer?: DecisionAnswer): number | undefined {
  const v = answer?.noul;
  return typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1
    ? v
    : undefined;
}

/** True only for a real, finite confidence in [0, 1] at or above the floor. */
function confident(answer: DecisionAnswer | undefined, floor: number): boolean {
  const c = answer?.confidence;
  return (
    typeof c === "number" &&
    Number.isFinite(c) &&
    c >= 0 &&
    c <= 1 &&
    c >= floor
  );
}

/**
 * Derive a conservative route from the answers. The gate only recommends a fast
 * route when the relevant numbers are decisive; otherwise it says "unknown" so
 * the caller falls back to the normal path.
 */
export function deriveRoute(
  answers: Record<string, DecisionAnswer>,
): TurnRoute {
  const needsTool = readNoul(answers["needs_tool"]);
  const fromContext = readNoul(answers["answerable_from_context"]);

  // Decisive "needs a tool" wins first — the harness must not skip work.
  if (needsTool !== undefined && needsTool >= 0.85) return "needs_tool";

  // Strongly answerable from context, with a present and low tool-need signal.
  if (
    fromContext !== undefined &&
    fromContext >= 0.85 &&
    needsTool !== undefined &&
    needsTool <= 0.3
  ) {
    return "answer_from_context";
  }

  // Clearly chat, with a calibrated intent and a present, low tool-need signal.
  if (
    answers["intent"]?.choice === "chat" &&
    confident(answers["intent"], PLAIN_CHAT_CONFIDENCE_FLOOR) &&
    needsTool !== undefined &&
    needsTool <= 0.2
  ) {
    return "plain_chat";
  }

  return "unknown";
}

/** Hostnames that refer to this machine. */
const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);

/** True when the endpoint points at this machine (loopback). */
function isLoopbackEndpoint(endpoint: string): boolean {
  try {
    const host = new URL(endpoint).hostname.toLowerCase();
    return LOOPBACK_HOSTS.has(host) || host.endsWith(".localhost");
  } catch {
    return false;
  }
}

/**
 * Issue one System One request. Returns the parsed answers on success, or null on
 * any failure (fail-open). Never throws.
 */
async function querySystemOne(
  state: string,
  questions: Record<string, QuestionSpec>,
  settings: DecisionGateSettings,
): Promise<Record<string, DecisionAnswer> | null> {
  try {
    const config: AxiosRequestConfig = {
      timeout: settings.timeoutMs,
      headers: { "Content-Type": "application/json" },
    };
    // The decision backend is a local server. Never tunnel the request through a
    // proxy, which cannot reach this machine and would make the gate useless.
    if (isLoopbackEndpoint(settings.endpoint)) {
      config.proxy = false;
    }
    const res = await axios.post<SystemOneResponse>(
      settings.endpoint,
      { model: settings.model, state, questions },
      config,
    );
    const data = res.data;
    if (!data || typeof data.answers !== "object" || data.answers === null) {
      return null;
    }
    return data.answers;
  } catch {
    // Any error, timeout, or non-2xx response fails open with no opinion.
    return null;
  }
}

/**
 * Classify a user turn through the gate. Returns null when the gate is disabled or
 * unavailable (fail-open). Otherwise returns a TurnDecision whose route is a
 * conservative recommendation the caller may act on in a later, validated step.
 */
export async function classifyTurn(
  userMessage: string,
  settings: DecisionGateSettings = resolveGateSettings(),
): Promise<TurnDecision | null> {
  if (!settings.enabled) return null;

  const startedAt = Date.now();
  const answers = await querySystemOne(userMessage, TURN_QUESTIONS, settings);
  if (!answers) return null;

  return { route: deriveRoute(answers), latencyMs: Date.now() - startedAt };
}

/** Extract the plain text from a user send request for classification. */
function extractUserText(message: SendMessageRequest): string {
  return message.content
    .filter(
      (part: MessageContent): part is MessageTextContent =>
        part.type === "text",
    )
    .map((part) => part.text)
    .join("\n")
    .trim();
}

/**
 * Shadow-mode hook. Fire-and-forget: classifies the outgoing user message and logs
 * the recommendation and latency through the project logger. It never awaits into
 * the caller, never throws, and never changes what gets sent, so it is safe to
 * call on the hot send path.
 */
export function shadowLogTurn(message: SendMessageRequest): void {
  const settings = resolveGateSettings();
  if (!settings.enabled) return;

  const text = extractUserText(message);
  if (!text) return;

  void classifyTurn(text, settings)
    .then((decision) => {
      if (decision) {
        EventLogger.warning(
          `[superfast] shadow route=${decision.route} latencyMs=${decision.latencyMs}`,
        );
      }
    })
    .catch(() => {
      // Fail open: a shadow measurement must never surface an error.
    });
}
