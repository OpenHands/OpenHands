/**
 * Renderer-side view of the DigitalOcean MARS bridge exposed by
 * `electron/preload-main.cjs`, plus the helpers that turn a connected MARS
 * session into an ordinary local backend.
 *
 * A session is reached over its public ingress URL when it has one (the main
 * process stamps the DigitalOcean token onto this renderer's requests to that
 * host, WebSocket handshake included), or over a port-forward tunnel on
 * loopback for sessions that cannot. Either way the far end is a plain
 * agent-server, so the backend is `kind: "local"`.
 *
 * `window.marsBridge` only exists in the Electron desktop build: harness-api
 * sends no CORS headers and the token must live in a main process, so the
 * browser and library builds cannot offer MARS at all. Callers gate on
 * `getMarsBridge()` rather than catching per-call failures.
 */

import { ConversationSortOrder } from "@openhands/typescript-client";
import { ConversationClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { validateLocalBackend } from "#/api/agent-server-compatibility";
import type { Backend } from "#/api/backend-registry/types";

export type MarsCredentialKind = "oauth" | "pat";

/** Non-secret description of a stored credential. Never carries the token. */
export interface MarsConnection {
  id: string;
  kind: MarsCredentialKind;
  label: string;
  teamName: string | null;
  /** ISO timestamp; OAuth grants expire after 30 days and cannot be refreshed. */
  expiresAt: string | null;
  isExpired: boolean;
}

export interface MarsAuthState {
  connections: MarsConnection[];
  active: MarsConnection | null;
  /** False when the OS keychain is unavailable, so tokens are session-only. */
  isPersistent: boolean;
  /** False when this build has no registered OAuth client; PAT still works. */
  canUseOAuth: boolean;
}

export interface MarsSession {
  session_id: string;
  name?: string | null;
  status?: string | null;
  agent_kind?: string | null;
  config_id?: string | null;
  repo_hint?: string | null;
  created_at?: string | null;
  last_event_at?: string | null;
}

export interface MarsSessionPage {
  sessions: MarsSession[];
  nextPageToken: string | null;
}

/**
 * Durable agent definition — what a user calls "an agent". One config has
 * many sessions. Not to be confused with `MarsSession.agent_kind`, which is
 * just an enum naming the coding agent inside the sandbox.
 */
export interface MarsAgentConfig {
  id: string;
  name?: string | null;
  created_by?: string | null;
  updated_at?: string | null;
  /**
   * The manifest's `agent` (e.g. `openhands`), lowercased. Filled in by the
   * main process; null when the manifest could not be read.
   */
  agent?: string | null;
}

const OPENHANDS_AGENT = "openhands";

/**
 * Whether a config's manifest declares `agent: openhands`. Manifests without
 * it are older specs that no longer launch, so an unknown agent never counts.
 */
export function isOpenHandsConfig(config: MarsAgentConfig): boolean {
  return config.agent === OPENHANDS_AGENT;
}

export interface NewMarsAgentInput {
  name: string;
  /** Stored as the agent's `OPENHANDS_LLM_API_KEY` DigitalOcean secret. */
  llmApiKey?: string;
}

/** harness-api's Agent Config name rule. */
const AGENT_NAME_PATTERN = /^[A-Za-z0-9]([A-Za-z0-9._-]{0,62}[A-Za-z0-9])?$/;

export function isValidAgentName(name: string): boolean {
  return AGENT_NAME_PATTERN.test(name);
}

export interface MarsAgentConfigPage {
  configs: MarsAgentConfig[];
  nextPageToken: string | null;
}

/** Most recent failed upstream dial behind an otherwise healthy listener. */
export interface MarsUpstreamFailure {
  /** harness-api WebSocket close code (4001 / 4002 / 4011), if any. */
  closeCode: number | null;
  /** HTTP status when the upgrade itself was refused. */
  httpStatus: number | null;
  message: string;
}

/**
 * How a connected session is reached: directly at its public ingress URL, or
 * through a loopback port-forward tunnel for sessions that cannot have one.
 */
export type MarsTransport = "ingress" | "tunnel";

export interface MarsTunnelStatus {
  sessionId: string;
  status: "connecting" | "connected" | "error";
  transport?: MarsTransport;
  /** Base URL the renderer registers as the backend host. */
  host?: string;
  /** harness-api's id for the ingress URL; absent over the tunnel. */
  ingressUrlId?: string;
  remotePort: number;
  /** Loopback listener port; only set over the tunnel. */
  localPort: number | undefined;
  error: string | undefined;
  upstreamFailure?: MarsUpstreamFailure | null;
}

export interface OpenMarsTunnelParams {
  sessionId: string;
  /** Re-request a specific local port, e.g. to revive a persisted backend. */
  localPort?: number;
}

interface ListOptions {
  pageSize?: number;
  pageToken?: string;
  status?: string;
}

export interface MarsBridge {
  getAuthState: () => Promise<MarsAuthState>;
  signInWithOAuth: () => Promise<MarsAuthState>;
  savePat: (payload: {
    token: string;
    label?: string;
  }) => Promise<MarsAuthState>;
  setActiveConnection: (id: string) => Promise<MarsAuthState>;
  signOut: (id?: string) => Promise<MarsAuthState>;
  listSessions: (options?: ListOptions) => Promise<MarsSessionPage>;
  listAgentConfigs: (
    options?: Omit<ListOptions, "status">,
  ) => Promise<MarsAgentConfigPage>;
  /** Empty means "no sessions yet", not "config is gone". */
  listConfigSessions: (
    configId: string,
    options?: ListOptions,
  ) => Promise<MarsSessionPage>;
  createOpenHandsAgent: (input: NewMarsAgentInput) => Promise<MarsAgentConfig>;
  /** Resolves once the new session is READY. */
  createSession: (configId: string, name: string) => Promise<MarsSession>;
  pauseSession: (sessionId: string) => Promise<void>;
  resumeSession: (sessionId: string) => Promise<void>;
  openTunnel: (params: OpenMarsTunnelParams) => Promise<MarsTunnelStatus>;
  closeTunnel: (sessionId: string) => Promise<void>;
  getTunnel: (sessionId: string) => Promise<MarsTunnelStatus | undefined>;
}

declare global {
  interface Window {
    marsBridge?: MarsBridge;
  }
}

export function getMarsBridge(): MarsBridge | null {
  if (typeof window === "undefined") return null;
  return window.marsBridge ?? null;
}

function requireMarsBridge(): MarsBridge {
  const bridge = getMarsBridge();
  if (!bridge) {
    throw new Error(
      "window.marsBridge is unavailable — MARS tunnels only work in the Electron desktop build.",
    );
  }
  return bridge;
}

/**
 * Connects to one MARS session — public ingress first, tunnel otherwise —
 * waking it if paused. The ingress hostname is revoked on pause and lock and
 * changes after rollback, so this always re-resolves rather than reusing a
 * host the renderer remembered.
 */
export function openMarsTunnel(
  params: OpenMarsTunnelParams,
): Promise<MarsTunnelStatus> {
  return requireMarsBridge().openTunnel(params);
}

/** Forgets the connection for one MARS session (closing its tunnel, if any). */
export function closeMarsTunnel(sessionId: string): Promise<void> {
  return requireMarsBridge().closeTunnel(sessionId);
}

const IPC_ERROR_PREFIX =
  /^Error invoking remote method '[^']+': (?:\w*Error: )?/;

/**
 * Message for a failed bridge call, without the wrapper Electron puts around
 * errors thrown in the main process.
 */
export function getMarsErrorMessage(error: unknown): string | null {
  const message = error instanceof Error ? error.message : null;
  return message ? message.replace(IPC_ERROR_PREFIX, "") : null;
}

export const SESSION_STATUS_READY = "SESSION_STATUS_READY";
export const SESSION_STATUS_PAUSED = "SESSION_STATUS_PAUSED";

/** Statuses a session can never come back from. */
const TERMINAL_SESSION_STATUSES = new Set([
  "SESSION_STATUS_DESTROYING",
  "SESSION_STATUS_DESTROYED",
  "SESSION_STATUS_FAILED",
]);

export type MarsSessionPhase =
  | "ready"
  | "paused"
  | "starting"
  | "ended"
  | "failed";

/**
 * Collapse `SESSION_STATUS_*` into the few states the UI treats differently.
 * Statuses are added server-side without a client release, so anything
 * unrecognised that is not terminal reads as "starting" (transitional).
 */
export function getSessionPhase(
  status: string | null | undefined,
): MarsSessionPhase {
  if (status === SESSION_STATUS_READY) return "ready";
  if (status === SESSION_STATUS_PAUSED) return "paused";
  if (status === "SESSION_STATUS_FAILED") return "failed";
  if (TERMINAL_SESSION_STATUSES.has(status ?? "")) return "ended";
  return "starting";
}

/** READY or PAUSED — connecting resumes a paused session first. */
export function isConnectableSession(session: MarsSession): boolean {
  const phase = getSessionPhase(session.status);
  return phase === "ready" || phase === "paused";
}

const SESSION_NAME_MAX_STEM = 32;

/**
 * Name for a session launched from an agent: the agent it came from plus a
 * suffix to keep siblings apart, narrowed to the character set MARS's own
 * generated names use.
 */
export function buildNewSessionName(configName: string | null | undefined) {
  const stem =
    (configName ?? "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, SESSION_NAME_MAX_STEM) || "agent";
  return `${stem}-${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * The in-guest agent-server's session key is set by the sandbox template, not
 * by us. The DigitalOcean token is the authorization boundary instead: the
 * ingress gateway requires it on every request (the main process injects it),
 * and dialing the tunnel requires it too.
 */
export const MARS_GUEST_API_KEY = "";

/**
 * An idle-paused session cold-wakes on first traffic and the in-guest
 * agent-server needs a moment more before it answers, so early probes
 * legitimately fail.
 */
const AGENT_SERVER_WAIT_MS = 60_000;
const AGENT_SERVER_PROBE_TIMEOUT_MS = 5_000;
const AGENT_SERVER_PROBE_INTERVAL_MS = 2_000;
/**
 * A just-woken sandbox also closes 4002 while its agent-server boots, so only
 * a 4002 that outlasts this window means nothing will ever serve the port.
 */
const GUEST_PORT_CLOSED_GRACE_MS = 20_000;

const CLOSE_SESSION_UNAVAILABLE = 4001;
const CLOSE_GUEST_DIAL_FAILED = 4002;
const HTTP_UNAUTHORIZED = 401;
const HTTP_FORBIDDEN = 403;

export type MarsAttachFailureReason =
  /** The sandbox answers, but not with an OpenHands agent-server on :8000. */
  | "not-openhands"
  /** DigitalOcean refused the tunnel for this token or session. */
  | "refused";

/** A failure the UI can explain precisely instead of a generic timeout. */
export class MarsAttachError extends Error {
  readonly reason: MarsAttachFailureReason;

  constructor(reason: MarsAttachFailureReason, message: string) {
    super(message);
    this.name = "MarsAttachError";
    this.reason = reason;
  }
}

function isRefusal(failure: MarsUpstreamFailure): boolean {
  return (
    failure.closeCode === CLOSE_SESSION_UNAVAILABLE ||
    failure.httpStatus === HTTP_UNAUTHORIZED ||
    failure.httpStatus === HTTP_FORBIDDEN
  );
}

async function readUpstreamFailure(
  sessionId: string | undefined,
): Promise<MarsUpstreamFailure | null> {
  const bridge = getMarsBridge();
  if (!sessionId || !bridge) return null;
  try {
    return (await bridge.getTunnel(sessionId))?.upstreamFailure ?? null;
  } catch {
    return null;
  }
}

/**
 * Poll a freshly opened tunnel until the agent-server behind it answers.
 * When `sessionId` is given, the tunnel's own upstream failures cut the wait
 * short with a {@link MarsAttachError}; otherwise the last probe failure is
 * rethrown so the user sees why it never came up.
 */
export async function waitForMarsAgentServer(
  host: string,
  {
    sessionId,
    budgetMs = AGENT_SERVER_WAIT_MS,
  }: { sessionId?: string; budgetMs?: number } = {},
): Promise<string | null> {
  const startedAt = Date.now();
  const deadline = startedAt + budgetMs;
  let guestPortClosedSince: number | null = null;
  for (;;) {
    try {
      return await validateLocalBackend(
        { host, apiKey: MARS_GUEST_API_KEY },
        AGENT_SERVER_PROBE_TIMEOUT_MS,
      );
    } catch (error) {
      const failure = await readUpstreamFailure(sessionId);
      if (failure && isRefusal(failure)) {
        throw new MarsAttachError("refused", failure.message);
      }
      const now = Date.now();
      if (failure?.closeCode === CLOSE_GUEST_DIAL_FAILED) {
        guestPortClosedSince ??= now;
        if (now - guestPortClosedSince >= GUEST_PORT_CLOSED_GRACE_MS) {
          throw new MarsAttachError("not-openhands", failure.message);
        }
      } else {
        guestPortClosedSince = null;
      }
      if (now >= deadline) throw error;
      await new Promise((resolve) => {
        setTimeout(resolve, AGENT_SERVER_PROBE_INTERVAL_MS);
      });
    }
  }
}

const LATEST_CONVERSATION_TIMEOUT_MS = 10_000;

/**
 * Id of the most recently updated conversation on a session's agent-server,
 * read through an explicit host override so it cannot race the registry
 * switching the active backend. Best-effort: null on any failure, because
 * the caller can always fall back to a fresh chat.
 */
export async function fetchLatestMarsConversationId(
  host: string,
): Promise<string | null> {
  try {
    const page = await new ConversationClient(
      getAgentServerClientOptions({
        host,
        sessionApiKey: MARS_GUEST_API_KEY,
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

export function buildMarsBackendHost(localPort: number): string {
  return `http://127.0.0.1:${localPort}`;
}

/**
 * The base URL a connection status says to register. Over ingress it is the
 * public URL; over the tunnel the loopback listener. A status from a bridge
 * that predates `host` is still honoured through its `localPort`.
 */
export function getMarsConnectionHost(
  status: MarsTunnelStatus,
): string | undefined {
  if (status.host) return status.host;
  return status.localPort === undefined
    ? undefined
    : buildMarsBackendHost(status.localPort);
}

/**
 * Local port a persisted MARS backend's tunnel was listening on, so a
 * restore can ask for it again. Undefined for an ingress-connected backend
 * (its host carries no explicit port), which is right: the public URL is
 * re-resolved rather than reused.
 */
export function getMarsBackendLocalPort(backend: Backend): number | undefined {
  try {
    const url = new URL(backend.host);
    if (url.hostname !== "127.0.0.1") return undefined;
    const port = Number.parseInt(url.port, 10);
    return Number.isInteger(port) && port > 0 ? port : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Build the Backend record for a session that now answers at `host`. The far
 * end really is an ordinary agent-server, so this is a `kind: "local"`
 * backend like any other, whether `host` is a public ingress URL or a
 * loopback tunnel.
 */
export function buildMarsBackendInput({
  name,
  host,
  sessionId,
  configId,
  apiKey = MARS_GUEST_API_KEY,
}: {
  name: string;
  host: string;
  sessionId: string;
  configId?: string;
  apiKey?: string;
}): Omit<Backend, "id" | "connectionRevision"> {
  return {
    name,
    host,
    apiKey,
    kind: "local",
    authMode: "api-key",
    marsSessionId: sessionId,
    ...(configId ? { marsConfigId: configId } : {}),
  };
}
