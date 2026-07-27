import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { isSdkHttpError } from "#/api/agent-server-compatibility";
import BashService from "#/api/bash-service/bash-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import type { SandboxStatus } from "#/api/conversation-service/agent-server-conversation-service.types";
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
  bashCommandId: string | null | undefined;
  enabled?: boolean;
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
 * - **Cloud backend**: pre-checks `sandbox_status` and the existence of
 *   a `conversation_url` before firing — paused, starting, errored, or
 *   missing sandboxes report a `sandboxIssue` and skip the request
 *   entirely (saves a doomed round-trip and gives the UI a targeted
 *   empty state). If the request does fire and fails with a 5xx /
 *   network error / 404 we re-classify it as `unreachable`.
 */
export function useBashCommandLogs(options: UseBashCommandLogsOptions) {
  const { conversationId, bashCommandId, enabled = true } = options;
  const active = useActiveBackend();
  // Only resolve the conversation when the modal is open. RunLogsModal mounts
  // (closed) for every activity-log row, so an unconditional lookup would fire
  // one /api/conversations request per row on page load. Passing null when
  // disabled trips useUserConversation's own `!!cid` gate.
  const conversationQuery = useUserConversation(
    enabled ? (conversationId ?? null) : null,
  );
  const conversation = conversationQuery.data;
  const conversationUrl = conversation?.conversation_url ?? null;
  const sessionApiKey = conversation?.session_api_key ?? null;

  const isCloud = active.backend.kind === "cloud";
  const conversationFetched = conversationQuery.isFetched;

  // Resolve a single "sandbox issue" only for cloud backends. Local
  // backends don't carry sandbox_status, and the agent-server hosts
  // events under a single root so there's nothing to gate on.
  let preflightIssue: SandboxIssue | null = null;
  let conversationMissing = false;
  if (isCloud && conversationFetched) {
    if (!conversation) {
      conversationMissing = true;
    } else {
      preflightIssue =
        sandboxIssueFromStatus(conversation.sandbox_status) ??
        (!conversation.conversation_url ? "missing" : null);
    }
  }

  // Cloud needs the conversation URL before it can talk to the
  // runtime; local does not.
  const hasRequiredAuth = isCloud ? !!conversationUrl : true;
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
      conversationUrl,
      sessionApiKey,
      active.backend.id,
      active.orgId,
    ],
    queryFn: () =>
      BashService.listOutputs(
        conversationUrl,
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
    /** True while we're still resolving the conversation runtime URL. */
    isResolvingConversation: isCloud && conversationQuery.isPending,
    /** Cloud-only: conversation lookup failed (deleted or no access). */
    conversationMissing,
    /**
     * Reason the bash query couldn't / didn't usefully complete. Always
     * null for healthy cloud sandboxes and for local backends.
     */
    sandboxIssue,
  };
}
