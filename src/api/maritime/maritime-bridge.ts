/**
 * Renderer-side view of the Maritime bridge exposed by
 * `electron/preload-main.cjs` (or, in the web build, by the Agent Canvas
 * server through src/api/maritime/maritime-web-bridge.ts), plus the helpers
 * that turn an open Maritime agent into an ordinary local backend.
 *
 * Each OpenHands agent on Maritime is a micro-VM running the agent-server,
 * reachable at its own HTTPS hostname. The browser talks to that hostname
 * directly with the agent-server's session key, so the backend is
 * `kind: "local"` like any remote agent-server. The Maritime API key stays in
 * the trusted process; a page with no bridge cannot offer Maritime at all, so
 * callers gate on `getMaritimeBridge()`.
 */

import { ConversationSortOrder } from "@openhands/typescript-client";
import { ConversationClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import {
  isDefinitiveLocalBackendError,
  validateLocalBackend,
} from "#/api/agent-server-compatibility";
import type { Backend } from "#/api/backend-registry/types";

/** Where the bridge keeps the API key, so the UI can say whether it lasts. */
export interface MaritimeStorage {
  kind: "keychain" | "encrypted-file" | "memory";
  label: string;
  /** The machine that holds the key: the one running Canvas. */
  host: string;
}

export interface MaritimeAuthState {
  isSignedIn: boolean;
  /** False when nothing on the host keeps the key, so it lasts this run only. */
  isSaved: boolean;
  storage: MaritimeStorage;
}

/** The account's plan and the memory range a new agent may use. */
export interface MaritimeAccount {
  plan: string | null;
  machines: number;
  includedMachines: number | null;
  awakeMachines: number;
  maxAwakeMachines: number | null;
  minMemMb: number;
  maxMemMb: number;
}

/** One OpenHands agent on Maritime. Never carries a key. */
export interface MaritimeAgent {
  id: string;
  name: string;
  /** Maritime's own status: active, sleeping, deploying, error or stopped. */
  status: string | null;
  webUrl: string | null;
  memMb: number | null;
  vcpus: number | null;
  createdAt: string | null;
  lastActiveAt: string | null;
  /** Whether this Canvas host holds the agent-server's session key. */
  hasAccessKey: boolean;
}

export interface NewMaritimeAgentInput {
  name: string;
  memMb: number;
  /** This page's origin, added to the agent-server's CORS list when needed. */
  origin?: string;
}

/** What opening an agent returns: everything a backend record needs. */
export interface MaritimeConnection {
  agentId: string;
  name: string;
  host: string;
  sessionApiKey: string;
}

export interface MaritimeBridge {
  getAuthState: () => Promise<MaritimeAuthState>;
  saveApiKey: (payload: { apiKey: string }) => Promise<MaritimeAuthState>;
  signOut: () => Promise<MaritimeAuthState>;
  getAccount: () => Promise<MaritimeAccount>;
  listAgents: () => Promise<MaritimeAgent[]>;
  /** Resolves once the new agent-server's VM is running. */
  createAgent: (input: NewMaritimeAgentInput) => Promise<MaritimeConnection>;
  /** Wakes the agent if it sleeps. */
  openAgent: (agentId: string) => Promise<MaritimeConnection>;
  /** Sets a new session key and restarts the agent-server. */
  resetAccessKey: (agentId: string) => Promise<MaritimeConnection>;
  sleepAgent: (agentId: string) => Promise<MaritimeAgent>;
  deleteAgent: (agentId: string) => Promise<void>;
}

declare global {
  interface Window {
    maritimeBridge?: MaritimeBridge;
  }
}

export function getMaritimeBridge(): MaritimeBridge | null {
  if (typeof window === "undefined") return null;
  return window.maritimeBridge ?? null;
}

export function requireMaritimeBridge(): MaritimeBridge {
  const bridge = getMaritimeBridge();
  if (!bridge) {
    throw new Error(
      "window.maritimeBridge is unavailable: Maritime needs the Electron desktop build or an Agent Canvas server that hosts the Maritime bridge.",
    );
  }
  return bridge;
}

const IPC_ERROR_PREFIX =
  /^Error invoking remote method '[^']+': (?:\w*Error: )?/;

/**
 * Message for a failed bridge call, without the wrapper Electron puts around
 * errors thrown in the main process. The bridge writes these for people.
 */
export function getMaritimeErrorMessage(error: unknown): string | null {
  const message = error instanceof Error ? error.message : null;
  return message ? message.replace(IPC_ERROR_PREFIX, "") : null;
}

export type MaritimeAgentPhase =
  | "running"
  | "sleeping"
  | "starting"
  | "stopped"
  | "failed";

/**
 * Collapse Maritime's statuses into the states the UI treats differently.
 * Anything unrecognised reads as "starting" (transitional).
 */
export function getMaritimeAgentPhase(
  status: string | null | undefined,
): MaritimeAgentPhase {
  switch (status) {
    case "active":
      return "running";
    case "sleeping":
      return "sleeping";
    case "stopped":
      return "stopped";
    case "error":
      return "failed";
    default:
      return "starting";
  }
}

/** Running, sleeping and stopped agents can be opened; opening wakes them. */
export function isOpenableAgent(agent: MaritimeAgent): boolean {
  const phase = getMaritimeAgentPhase(agent.status);
  return phase === "running" || phase === "sleeping" || phase === "stopped";
}

/** Memory sizes offered for a new agent, in MB, inside the account's range. */
const MEMORY_STEPS_MB = [2048, 4096, 8192];
/** The agent-server with its browser tools is comfortable at 4 GB. */
const PREFERRED_MEMORY_MB = 4096;

export function getMemoryOptions(account: MaritimeAccount): number[] {
  const options = MEMORY_STEPS_MB.filter(
    (mb) => mb >= account.minMemMb && mb <= account.maxMemMb,
  );
  return options.length ? options : [account.maxMemMb];
}

export function getDefaultMemory(account: MaritimeAccount): number {
  const options = getMemoryOptions(account);
  return options.includes(PREFERRED_MEMORY_MB)
    ? PREFERRED_MEMORY_MB
    : options[options.length - 1];
}

/** Maritime's agent name rule as the bridge enforces it. */
const AGENT_NAME_PATTERN = /^[A-Za-z0-9]([A-Za-z0-9._-]{0,62}[A-Za-z0-9])?$/;

export function isValidMaritimeAgentName(name: string): boolean {
  return AGENT_NAME_PATTERN.test(name);
}

/** A name for a new agent that will not clash with the user's others. */
export function buildDefaultAgentName(): string {
  return `openhands-${Math.random().toString(36).slice(2, 8)}`;
}

/** Where a freshly opened agent lands when it has no conversation yet. */
export const NEW_CHAT_PATH = "/conversations";

/**
 * A just-created or just-woken VM answers before its agent-server has bound
 * its port, so early probes legitimately fail.
 */
const AGENT_SERVER_WAIT_MS = 120_000;
const AGENT_SERVER_PROBE_TIMEOUT_MS = 5_000;
const AGENT_SERVER_PROBE_INTERVAL_MS = 2_000;

/**
 * Poll the agent-server until it answers, through the same check the Add
 * Backend form runs (settings and server info, with the session key). A
 * rejected key or unsupported version is rethrown at once; otherwise the
 * last failure is rethrown when the budget runs out.
 */
export async function waitForMaritimeAgentServer(
  connection: Pick<MaritimeConnection, "host" | "sessionApiKey">,
  { budgetMs = AGENT_SERVER_WAIT_MS }: { budgetMs?: number } = {},
): Promise<string | null> {
  const deadline = Date.now() + budgetMs;
  for (;;) {
    try {
      return await validateLocalBackend(
        { host: connection.host, apiKey: connection.sessionApiKey },
        AGENT_SERVER_PROBE_TIMEOUT_MS,
      );
    } catch (error) {
      if (isDefinitiveLocalBackendError(error)) throw error;
      if (Date.now() >= deadline) throw error;
      await new Promise((resolve) => {
        setTimeout(resolve, AGENT_SERVER_PROBE_INTERVAL_MS);
      });
    }
  }
}

const LATEST_CONVERSATION_TIMEOUT_MS = 10_000;

/**
 * Id of the most recently updated conversation on the agent's server, read
 * through an explicit host so it cannot race the registry switching the
 * active backend. Best-effort: null on any failure.
 */
export async function fetchLatestMaritimeConversationId(
  connection: Pick<MaritimeConnection, "host" | "sessionApiKey">,
): Promise<string | null> {
  try {
    const page = await new ConversationClient(
      getAgentServerClientOptions({
        host: connection.host,
        sessionApiKey: connection.sessionApiKey,
        timeout: LATEST_CONVERSATION_TIMEOUT_MS,
      }),
    ).searchConversations({
      limit: 1,
      sort_order: ConversationSortOrder.UPDATED_AT_DESC,
    });
    return page?.items?.[0]?.id ?? null;
  } catch {
    return null;
  }
}

/**
 * The Backend record for an open agent. The far end is an ordinary
 * agent-server, so this is a `kind: "local"` backend like any other.
 */
export function buildMaritimeBackendInput(
  connection: MaritimeConnection,
): Omit<Backend, "id" | "connectionRevision"> {
  return {
    name: connection.name,
    host: connection.host,
    apiKey: connection.sessionApiKey,
    kind: "local",
    authMode: "api-key",
    maritimeAgentId: connection.agentId,
  };
}
