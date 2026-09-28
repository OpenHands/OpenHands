import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { isSdkHttpError } from "#/api/agent-server-compatibility";
import BashService from "#/api/bash-service/bash-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import type { SandboxStatus } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useCloudSandbox } from "./use-cloud-sandbox";
import { useUserConversation } from "./use-user-conversation";

export const BASH_COMMAND_LOGS_QUERY_KEY = ["bash-command-logs"] as const;

/**
 * Reasons the modal can't fetch logs from a cloud sandbox, in priority
 * order. The hook surfaces at most one of these so the UI can render a
 * targeted message instead of a raw error.
 */
export type SandboxIssue =
  | "missing" // sandbox has been deleted (or conversation has no runtime URL)
  | "paused" // sandbox is paused — needs resuming
  | "starting" // sandbox is still booting
  | "errored" // sandbox is in a terminal error state
  | "unreachable"; // bash query attempted and failed at the network layer

interface UseBashCommandLogsOptions {
  /**
   * The agent-server conversation that hosts the bash command. Used to
   * resolve `conversation_url` and `session_api_key` for cloud
   * backends, and to gate the query on `sandbox_status` so we don't
   * fire requests at known-unreachable sandboxes.
   */
  conversationId: string | null | undefined;
  /**
   * The cloud sandbox that ran the command. Script automations never
   * create a conversation, so on cloud backends this is the only handle
   * to their agent-server: its `AGENT_SERVER` exposed URL and
   * `session_api_key` stand in for the conversation's `conversation_url`
   * and `session_api_key`. Ignored when a conversation id is given, and
   * on local backends.
   */
  sandboxId?: string | null;
  bashCommandId: string | null | undefined;
  enabled?: boolean;
}

/** Name of the agent-server entry in a cloud sandbox's `exposed_urls`. */
const AGENT_SERVER_EXPOSED_URL_NAME = "AGENT_SERVER";

/** The agent-server that holds the command's events, however it was found. */
interface RuntimeTarget {
  url: string | null;
  sessionApiKey: string | null;
  sandboxStatus: SandboxStatus | null;
}

/**
 * Map a cloud sandbox status to a stable issue code (or null when the
 * sandbox is healthy enough to attempt the fetch).
 */
function sandboxIssueFromStatus(
  status: SandboxStatus | null | undefined,
): SandboxIssue | null {
  switch (status) {
    case "MISSING":
      return "missing";
    case "PAUSED":
      return "paused";
    case "STARTING":
      return "starting";
    case "ERROR":
      return "errored";
    case "RUNNING":
    case null:
    case undefined:
    default:
      return null;
  }
}

/**
 * Detect "the runtime is unreachable" errors from the cloud proxy. The
 * proxy itself returns 5xx when the upstream sandbox is gone; runtimes
 * return 4xx/5xx for various ephemeral states. We classify 5xx and
 * network errors as "unreachable" so the modal can render the
 * sandbox-gone state instead of dumping a raw error. Cloud calls go
 * through the shared TypeScript client and throw its `HttpError`;
 * axios-shaped errors are still recognized as well.
 */
function classifyFetchError(error: unknown): SandboxIssue | null {
  const status = axios.isAxiosError(error)
    ? error.response?.status
    : isSdkHttpError(error)
      ? (error as { status: number }).status
      : undefined;
  if (status !== undefined) {
    // Treat 502/503/504 (proxy can't reach upstream) and 404 (sandbox or
    // resource no longer exists) as the sandbox being gone. We do not
    // collapse 401/403 here — those are auth bugs we want to surface.
    return status === 404 || status >= 500 ? "unreachable" : null;
  }
  // No status → the request never got a response. Axios reports these as
  // response-less errors; fetch (the shared client) throws `TypeError`
  // for network failures and `AbortError`/`TimeoutError` for timeouts,
  // sometimes wrapped in a plain `Error` with the original as `cause`.
  if (axios.isAxiosError(error) || error instanceof TypeError) {
    return "unreachable";
  }
  if (error instanceof Error) {
    const causeName = error.cause instanceof Error ? error.cause.name : null;
    if (
      error.name === "AbortError" ||
      error.name === "TimeoutError" ||
      causeName === "AbortError" ||
      causeName === "TimeoutError"
    ) {
      return "unreachable";
    }
  }
  return null;
}

/**
 * Search `BashOutput` events for an automation run's bash command.
 *
 * - **Local backend**: the query fires as soon as the modal opens and
 *   we have a `bash_command_id`. The conversation lookup runs in
 *   parallel; if it resolves with `session_api_key`/`conversation_url`
 *   those are passed through, but a missing/stale conversation does not
 *   block the bash query (the local agent-server hosts events under a
 *   single root).
 * - **Cloud backend**: the run's agent-server is found through its
 *   conversation when it has one, and through its sandbox otherwise
 *   (script automations never create a conversation). Either lookup
 *   pre-checks the sandbox status and the existence of a runtime URL
 *   before firing — paused, starting, errored, or missing sandboxes
 *   report a `sandboxIssue` and skip the request entirely (saves a
 *   doomed round-trip and gives the UI a targeted empty state). A run
 *   with neither handle reports `missing` at once. If the request does
 *   fire and fails with a 5xx / network error / 404 we re-classify it
 *   as `unreachable`.
 */
export function useBashCommandLogs(options: UseBashCommandLogsOptions) {
  const { conversationId, sandboxId, bashCommandId, enabled = true } = options;
  const active = useActiveBackend();
  const isCloud = active.backend.kind === "cloud";

  // Only resolve the runtime when the modal is open. RunLogsModal mounts
  // (closed) for every activity-log row, so an unconditional lookup would fire
  // one /api/conversations request per row on page load. Passing null when
  // disabled trips each lookup hook's own `!!id` gate.
  const viaConversation = enabled && !!conversationId;
  const viaSandbox = enabled && isCloud && !viaConversation && !!sandboxId;
  const conversationQuery = useUserConversation(
    viaConversation ? (conversationId as string) : null,
  );
  const sandboxQuery = useCloudSandbox(viaSandbox ? sandboxId : null);
  const conversation = conversationQuery.data;

  let runtime: RuntimeTarget | null = null;
  let isResolvingRuntime = false;
  let conversationMissing = false;
  let preflightIssue: SandboxIssue | null = null;

  if (!isCloud) {
    // Local backends don't carry sandbox_status, and the agent-server hosts
    // events under a single root so there's nothing to gate on.
    runtime = {
      url: conversation?.conversation_url ?? null,
      sessionApiKey: conversation?.session_api_key ?? null,
      sandboxStatus: null,
    };
  } else if (viaConversation) {
    // `isLoading`, not `isPending`: a query that is disabled — including
    // when `useUserConversation` disables itself — stays `pending` forever
    // without ever fetching, and reporting that as "resolving" is what left
    // the modal loading with no request in flight.
    isResolvingRuntime = conversationQuery.isLoading;
    if (conversationQuery.isFetched) {
      if (!conversation) {
        conversationMissing = true;
      } else {
        runtime = {
          url: conversation.conversation_url ?? null,
          sessionApiKey: conversation.session_api_key ?? null,
          sandboxStatus: conversation.sandbox_status ?? null,
        };
      }
    }
  } else if (viaSandbox) {
    isResolvingRuntime = sandboxQuery.isLoading;
    if (sandboxQuery.isFetched) {
      const sandbox = sandboxQuery.data;
      if (sandboxQuery.isError) {
        preflightIssue = "unreachable";
      } else if (!sandbox) {
        // Deleted, or created by someone else: the lookup only returns the
        // caller's own sandboxes.
        preflightIssue = "missing";
      } else {
        runtime = {
          url:
            sandbox.exposed_urls?.find(
              (exposed) => exposed.name === AGENT_SERVER_EXPOSED_URL_NAME,
            )?.url ?? null,
          sessionApiKey: sandbox.session_api_key,
          sandboxStatus: sandbox.status,
        };
      }
    }
  } else if (enabled) {
    // Neither a conversation nor a sandbox to ask: nothing to load, and
    // nothing to wait for.
    preflightIssue = "missing";
  }

  if (isCloud && runtime && !preflightIssue) {
    preflightIssue =
      sandboxIssueFromStatus(runtime.sandboxStatus) ??
      (!runtime.url ? "missing" : null);
  }

  const runtimeUrl = runtime?.url ?? null;
  const sessionApiKey = runtime?.sessionApiKey ?? null;

  // Cloud needs the runtime URL before it can talk to the agent-server;
  // local does not.
  const hasRequiredAuth = isCloud ? !!runtimeUrl : true;
  const canFire =
    enabled &&
    !!bashCommandId &&
    hasRequiredAuth &&
    !preflightIssue &&
    !conversationMissing;

  const query = useQuery({
    queryKey: [
      ...BASH_COMMAND_LOGS_QUERY_KEY,
      bashCommandId,
      runtimeUrl,
      sessionApiKey,
      active.backend.id,
      active.orgId,
    ],
    queryFn: () =>
      BashService.listOutputs(
        runtimeUrl,
        sessionApiKey,
        bashCommandId as string,
      ),
    enabled: canFire,
    // Completed-run logs don't change — cache long enough that reopening
    // the modal is instant but not forever.
    staleTime: 60 * 1000,
    gcTime: 5 * 60 * 1000,
    retry: false,
  });

  // If the request fired and failed in a way that suggests the
  // sandbox is gone/unreachable, surface it as a sandbox issue so the
  // modal can render the matching empty state instead of a raw error.
  const fetchIssue = isCloud ? classifyFetchError(query.error) : null;
  const sandboxIssue: SandboxIssue | null = preflightIssue ?? fetchIssue;

  return {
    data: query.data,
    /**
     * Set only when the request actually fired and failed AND the
     * failure isn't already classified as a sandbox issue. The modal
     * should render `sandboxIssue` first and only fall back to this.
     */
    error: fetchIssue ? null : query.error,
    isFetching: query.isFetching,
    isPending: query.isPending,
    /**
     * True while we're still resolving the runtime URL through the
     * conversation or, failing that, the sandbox.
     */
    isResolvingConversation: isResolvingRuntime,
    /** Cloud-only: conversation lookup failed (deleted or no access). */
    conversationMissing,
    /**
     * Reason the bash query couldn't / didn't usefully complete. Always
     * null for healthy cloud sandboxes and for local backends.
     */
    sandboxIssue,
  };
}
