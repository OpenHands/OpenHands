import i18n from "#/i18n";
import { I18nKey } from "#/i18n/declaration";
import {
  compareAgentServerVersions,
  getCachedAgentServerVersion,
} from "#/api/agent-server-compatibility";
import {
  createCloudAppConversation,
  getCloudAppConversationStartTask,
  pickCloudBackendForLaunch,
} from "#/api/cloud/conversation-service.api";
import type { Backend } from "#/api/backend-registry/types";
import {
  getStoredConversationMetadata,
  type ConversationMetadata,
} from "#/api/conversation-metadata-store";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversationStartTask } from "#/api/conversation-service/agent-server-conversation-service.types";
import {
  CHILD_CONVERSATION_ISOLATIONS,
  CHILD_CONVERSATION_RESULT_PREFIX,
  CHILD_CONVERSATION_TARGETS,
  LAUNCH_CHILD_CONVERSATION_TOOL_NAME,
  MIN_AGENT_SERVER_VERSION_FOR_PARENT_LINK,
  type ChildConversationIsolation,
  type ChildConversationTarget,
} from "#/constants/child-conversation";
import { useGoalStore } from "#/stores/goal-store";
import type { LaunchChildConversationAction } from "#/types/agent-server/core";
import { buildAgentCanvasPath } from "#/utils/base-path";
import {
  displayErrorToast,
  displaySuccessToastWithLink,
} from "#/utils/custom-toast-handlers";

/** Cadence and ceiling for waiting on a Cloud sandbox to expose its id. */
const CLOUD_START_POLL_INTERVAL_MS = 3_000;
const CLOUD_START_POLL_TIMEOUT_MS = 180_000;

const LEDGER_STORAGE_KEY_PREFIX = "openhands-child-conversation-launches:";

interface LaunchSuccess {
  status: "launched";
  target: ChildConversationTarget;
  conversation_id: string | null;
  url: string | null;
  initial_status: string;
  title: string | null;
  /** Local only: the directory the child inherited from this conversation. */
  workspace?: string;
  /** Local only: which isolation mode was actually applied. */
  isolation?: ChildConversationIsolation;
  /**
   * Present when the applied isolation is not the one that was asked for,
   * explaining why and what it means for the child.
   */
  isolation_note?: string;
  /** Cloud only: poll this if `conversation_id` is still null. */
  start_task_id?: string;
  /** Cloud only: which connected Cloud backend the child was launched on. */
  backend?: string;
  /** Whether the parent/child link was persisted server-side. */
  parent_link: boolean;
  /** Present when `parent_link` is false, explaining why. */
  parent_link_note?: string;
}

interface LaunchFailure {
  status: "error";
  error: string;
  guidance: string;
}

export type LaunchChildConversationResult = LaunchSuccess | LaunchFailure;

interface ValidatedParams {
  target: ChildConversationTarget;
  task: string;
  title: string | null;
  repository: string | null;
  branch: string | null;
  isolation: ChildConversationIsolation;
}

const failure = (error: string, guidance: string): LaunchFailure => ({
  status: "error",
  error,
  guidance,
});

const quoted = (values: readonly string[]) =>
  values.map((value) => `"${value}"`).join(" or ");

const blankToNull = (value: string | null | undefined) => {
  const trimmed = value?.trim();
  return trimmed ? trimmed : null;
};

/**
 * Validate what the agent-server cannot.
 *
 * The tool's JSON Schema is turned into a pydantic model with `extra="forbid"`,
 * so unknown/misspelled parameter names, missing required parameters and wrong
 * types already fail server-side with a corrective message. `enum` is the gap:
 * the SDK advertises it to the LLM but drops it when building the model, so a
 * misspelled `target` reaches us intact and must be rejected here. The
 * cross-target rules (cloud-only / local-only parameters) are ours to enforce
 * too, since a schema cannot express them.
 */
function validateLaunchParams(
  action: LaunchChildConversationAction,
):
  | { ok: true; params: ValidatedParams }
  | { ok: false; failure: LaunchFailure } {
  const target = action.target as ChildConversationTarget;
  if (!CHILD_CONVERSATION_TARGETS.includes(target)) {
    return {
      ok: false,
      failure: failure(
        `Unknown target ${JSON.stringify(action.target)}.`,
        `\`target\` must be exactly ${quoted(CHILD_CONVERSATION_TARGETS)}. Nothing was launched — call ${LAUNCH_CHILD_CONVERSATION_TOOL_NAME} again with a valid target.`,
      ),
    };
  }

  const task = blankToNull(action.task);
  if (!task) {
    return {
      ok: false,
      failure: failure(
        "`task` is empty.",
        "`task` must be a self-contained brief: the child conversation cannot see this one, so state the goal, constraints and expected output.",
      ),
    };
  }

  const repository = blankToNull(action.repository);
  const branch = blankToNull(action.branch);
  const isolation = blankToNull(
    action.isolation,
  ) as ChildConversationIsolation | null;

  if (target === "local" && (repository || branch)) {
    return {
      ok: false,
      failure: failure(
        '`repository`/`branch` were passed with target="local".',
        'A local child always runs in this conversation\'s workspace. Drop `repository` and `branch`, or use target="cloud" to run against a repository in a Cloud sandbox.',
      ),
    };
  }

  if (target === "cloud" && isolation) {
    return {
      ok: false,
      failure: failure(
        '`isolation` was passed with target="cloud".',
        'Cloud children always run in their own isolated sandbox. Drop `isolation`, or use target="local" to choose between "worktree" and "shared".',
      ),
    };
  }

  if (branch && !repository) {
    return {
      ok: false,
      failure: failure(
        "`branch` was passed without `repository`.",
        'Pass `repository` as "owner/repo" alongside `branch`, or drop `branch` to use the repository\'s default branch.',
      ),
    };
  }

  if (isolation && !CHILD_CONVERSATION_ISOLATIONS.includes(isolation)) {
    return {
      ok: false,
      failure: failure(
        `Unknown isolation ${JSON.stringify(action.isolation)}.`,
        `\`isolation\` must be exactly ${quoted(CHILD_CONVERSATION_ISOLATIONS)}. Nothing was launched — call ${LAUNCH_CHILD_CONVERSATION_TOOL_NAME} again with a valid isolation.`,
      ),
    };
  }

  return {
    ok: true,
    params: {
      target,
      task,
      title: blankToNull(action.title),
      repository,
      branch,
      isolation: isolation ?? "worktree",
    },
  };
}

/**
 * Remember which tool calls have already been acted on.
 *
 * Unlike the Canvas UI tool, a launch is not idempotent: replaying its
 * ActionEvent (a socket reconnect that falls back to `resend_mode: "all"`, or a
 * REST/WebSocket race after a reload) would start a second — on Cloud, billable
 * — conversation. Claim the tool call before any network work so a replay that
 * arrives mid-flight is dropped too.
 */
function claimToolCall(parentConversationId: string, toolCallId: string) {
  const key = `${LEDGER_STORAGE_KEY_PREFIX}${parentConversationId}`;
  let handled: string[] = [];
  try {
    const raw = window.localStorage.getItem(key);
    const parsed: unknown = raw ? JSON.parse(raw) : null;
    if (Array.isArray(parsed)) {
      handled = parsed.filter((id): id is string => typeof id === "string");
    }
  } catch {
    // A corrupt ledger must not block the launch; start a fresh one.
  }

  if (handled.includes(toolCallId)) return false;

  try {
    window.localStorage.setItem(key, JSON.stringify([...handled, toolCallId]));
  } catch {
    // Storage full or unavailable — proceed, accepting replay risk over never
    // launching at all.
  }
  return true;
}

function absoluteCanvasUrl(path: string) {
  const canvasPath = buildAgentCanvasPath(path);
  if (typeof window === "undefined") return canvasPath;
  return new URL(canvasPath, window.location.origin).toString();
}

/**
 * `parent_conversation_id` on the local start request landed in agent-server
 * 1.37.1. Older servers ignore unknown fields, so the child is created without
 * the link — report that rather than letting the agent assume a relationship
 * that does not exist.
 */
function localParentLinkNote(): string | null {
  const version = getCachedAgentServerVersion();
  if (!version) return null;
  const comparison = compareAgentServerVersions(
    version,
    MIN_AGENT_SERVER_VERSION_FOR_PARENT_LINK,
  );
  if (comparison === null || comparison >= 0) return null;
  return `Agent server ${version} does not persist parent/child conversation links (needs ${MIN_AGENT_SERVER_VERSION_FOR_PARENT_LINK}); the child was created but is not linked to this conversation.`;
}

/**
 * Whether the parent's workspace can host a git worktree.
 *
 * A conversation started without a repository or an attached workspace runs in
 * a scratch directory that is `git init`-ed but never committed to. `git
 * worktree add` cannot branch from an unborn HEAD, and the agent-server raises
 * the resulting error straight out of its start-conversation handler as a 500,
 * so asking for a worktree there fails the whole launch. Conversation metadata
 * is only persisted when a repository or an explicit workspace was chosen (see
 * `createConversation`), so its absence pinpoints exactly those scratch
 * workspaces. Anything else is attempted and falls back on failure, because the
 * frontend cannot inspect the workspace's git state directly.
 */
const parentSupportsWorktree = (
  metadata: ConversationMetadata | null,
): boolean => !!(metadata?.selected_repository || metadata?.selected_workspace);

const SHARED_FALLBACK_CONSEQUENCE =
  "The child was launched in this conversation's directory instead, so it can see work in progress here and the two agents may conflict over the same files.";

async function launchLocalChild(
  params: ValidatedParams,
  parentConversationId: string,
): Promise<LaunchChildConversationResult> {
  // The agent-server rejects a parent whose workspace differs from the child's,
  // so the child requests the parent's own directory. `worktree` then carves an
  // isolated worktree out of it, which is what keeps siblings from colliding.
  const workspace =
    await AgentServerConversationService.resolveConversationWorkingDir(
      parentConversationId,
    );
  const parentMetadata = getStoredConversationMetadata(parentConversationId);

  const createChild = (isolation: ChildConversationIsolation) =>
    AgentServerConversationService.createConversation({
      initialUserMsg: params.task,
      metadata: parentMetadata
        ? {
            selected_repository: parentMetadata.selected_repository,
            selected_branch: parentMetadata.selected_branch,
            git_provider: parentMetadata.git_provider,
          }
        : null,
      workingDirOverride: workspace,
      workspaceMode: isolation === "shared" ? "local_repo" : "new_worktree",
      parentConversationId,
    });

  let isolation = params.isolation;
  let isolationNote: string | null = null;

  if (isolation === "worktree" && !parentSupportsWorktree(parentMetadata)) {
    isolation = "shared";
    isolationNote = `This conversation's workspace is a scratch directory with no commits, which git cannot cut a worktree from. ${SHARED_FALLBACK_CONSEQUENCE}`;
  }

  let startTask;
  try {
    startTask = await createChild(isolation);
  } catch (worktreeError) {
    if (isolation !== "worktree") throw worktreeError;
    // Creating the worktree is the one part of a local launch that can fail on
    // an otherwise healthy workspace, so retry without it rather than losing
    // the launch. Anything else fails both attempts and is reported as-is.
    try {
      startTask = await createChild("shared");
    } catch {
      throw worktreeError;
    }
    isolation = "shared";
    isolationNote = `Creating a git worktree in this conversation's workspace failed (${worktreeError instanceof Error ? worktreeError.message : String(worktreeError)}). ${SHARED_FALLBACK_CONSEQUENCE}`;
  }

  const conversationId = startTask.app_conversation_id ?? startTask.id;

  if (params.title) {
    // Local start requests carry no title — only `autotitle`. Best effort: a
    // failed rename must not fail the launch.
    await AgentServerConversationService.updateConversationTitle(
      conversationId,
      params.title,
    ).catch(() => undefined);
  }

  const parentLinkNote = localParentLinkNote();
  return {
    status: "launched",
    target: "local",
    conversation_id: conversationId,
    url: absoluteCanvasUrl(`/conversations/${conversationId}`),
    initial_status: startTask.status,
    title: params.title,
    workspace,
    isolation,
    ...(isolationNote ? { isolation_note: isolationNote } : {}),
    parent_link: !parentLinkNote,
    ...(parentLinkNote ? { parent_link_note: parentLinkNote } : {}),
  };
}

const delay = (ms: number) =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

/**
 * Wait for the Cloud start task to expose its conversation id.
 *
 * The Cloud API provisions a sandbox asynchronously and only fills in
 * `app_conversation_id` once the task reaches READY, so without this the agent
 * would get a result with no conversation to open. Bounded: on timeout the
 * caller reports the still-provisioning task instead of hanging.
 */
async function waitForCloudConversationId(
  task: AppConversationStartTask,
  backend: Backend,
): Promise<AppConversationStartTask> {
  const deadline = Date.now() + CLOUD_START_POLL_TIMEOUT_MS;
  let latest = task;
  while (!latest.app_conversation_id && latest.status !== "ERROR") {
    if (Date.now() >= deadline) break;

    await delay(CLOUD_START_POLL_INTERVAL_MS);

    const next = await getCloudAppConversationStartTask(
      latest.id,
      backend,
    ).catch(() => null);
    if (!next) break;
    latest = next;
  }
  return latest;
}

async function launchCloudChild(
  params: ValidatedParams,
  parentConversationId: string,
): Promise<LaunchChildConversationResult> {
  const backend = pickCloudBackendForLaunch();
  if (!backend) {
    return failure(
      "No OpenHands Cloud backend is connected in Agent Canvas.",
      'Ask the user to connect OpenHands Cloud from the backend picker, then call this tool again — or relaunch now with target="local".',
    );
  }

  const parentMetadata = getStoredConversationMetadata(parentConversationId);
  const startTask = await createCloudAppConversation(
    {
      initial_message: {
        role: "user",
        content: [{ type: "text", text: params.task }],
      },
      title: params.title,
      // Fall back to this conversation's repository so a bare `target: "cloud"`
      // call still lands the child on the code the user is working on. The
      // provider only travels with the inherited repository — it may not apply
      // to one the agent named itself.
      selected_repository:
        params.repository ?? parentMetadata?.selected_repository ?? null,
      selected_branch: params.branch ?? null,
      git_provider: params.repository
        ? null
        : (parentMetadata?.git_provider ?? null),
      // The parent runs on the local agent-server, so its id means nothing to
      // Cloud. Sending it would only hide the child from the Cloud
      // conversation list, which filters out anything with a parent.
      parent_conversation_id: null,
    },
    backend,
  );

  const settled = await waitForCloudConversationId(startTask, backend);
  if (settled.status === "ERROR") {
    return failure(
      settled.detail || "The Cloud conversation failed to start.",
      'The Cloud sandbox could not be provisioned. Report this to the user; you can retry, or fall back to target="local".',
    );
  }

  const conversationId = settled.app_conversation_id;
  return {
    status: "launched",
    target: "cloud",
    conversation_id: conversationId,
    url: conversationId
      ? `${backend.host.replace(/\/$/, "")}/conversations/${conversationId}`
      : null,
    initial_status: settled.status,
    title: params.title,
    start_task_id: settled.id,
    backend: backend.name,
    // Cloud children of a local parent carry no server-side link; see above.
    parent_link: false,
    parent_link_note:
      "This conversation runs on the local agent server, so OpenHands Cloud has no parent to link the child to.",
  };
}

/**
 * Hand the outcome back to the agent and the user.
 *
 * Client tools are acknowledged by the agent-server before the browser does any
 * work, so a message is the only way to give the agent the child's id or tell
 * it how to fix a malformed call. `sendMessage` runs the agent, which is what
 * makes it relay the result to the user in its next turn.
 */
async function reportLaunchResult(
  parentConversationId: string,
  result: LaunchChildConversationResult,
) {
  if (result.status === "error") {
    displayErrorToast(
      i18n.t(I18nKey.CHILD_CONVERSATION$LAUNCH_FAILED, {
        error: result.error,
      }),
    );
  } else if (result.url) {
    displaySuccessToastWithLink(
      i18n.t(
        result.target === "cloud"
          ? I18nKey.CHILD_CONVERSATION$LAUNCHED_CLOUD
          : I18nKey.CHILD_CONVERSATION$LAUNCHED_LOCAL,
      ),
      i18n.t(I18nKey.CHILD_CONVERSATION$OPEN),
      result.url,
    );
  }

  // The agent-server cancels an active `/goal` loop on any inbound message, so
  // a launch must not silently end one. The toast above still tells the user
  // what happened.
  const goalStatus =
    useGoalStore.getState().statusByConversation[parentConversationId];
  if (goalStatus?.active) return;

  await AgentServerConversationService.sendMessage(parentConversationId, {
    role: "user",
    content: [
      {
        type: "text",
        text: `${CHILD_CONVERSATION_RESULT_PREFIX}${JSON.stringify(result)}`,
      },
    ],
  });
}

/**
 * Execute a `launch_child_conversation` tool call.
 *
 * Never rejects: every failure is turned into corrective guidance for the
 * agent, because the agent-server has already told it the call succeeded.
 */
export async function handleLaunchChildConversationAction(
  action: LaunchChildConversationAction,
  parentConversationId: string,
  toolCallId: string,
): Promise<void> {
  if (!claimToolCall(parentConversationId, toolCallId)) return;

  const validation = validateLaunchParams(action);
  let result: LaunchChildConversationResult;
  if (!validation.ok) {
    result = validation.failure;
  } else {
    try {
      result =
        validation.params.target === "cloud"
          ? await launchCloudChild(validation.params, parentConversationId)
          : await launchLocalChild(validation.params, parentConversationId);
    } catch (error) {
      result = failure(
        error instanceof Error ? error.message : String(error),
        "The launch request failed. Report the error to the user; retry only if the cause looks transient.",
      );
    }
  }

  await reportLaunchResult(parentConversationId, result).catch((error) => {
    console.warn(
      `[${LAUNCH_CHILD_CONVERSATION_TOOL_NAME}] Failed to report the launch result:`,
      error,
    );
  });
}
