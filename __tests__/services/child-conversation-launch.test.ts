import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  CHILD_CONVERSATION_RESULT_PREFIX,
  LAUNCH_CHILD_CONVERSATION_ACTION_KIND,
  LAUNCH_CHILD_CONVERSATION_TOOL_NAME,
} from "#/constants/child-conversation";
import {
  getStoredConversationMetadata,
  setStoredConversationMetadata,
} from "#/api/conversation-metadata-store";
import { getQueryClient } from "#/query-client-config";
import { useGoalStore } from "#/stores/goal-store";
import type { LaunchChildConversationAction } from "#/types/agent-server/core";
import { isLaunchChildConversationActionEvent } from "#/types/agent-server/type-guards";
import {
  handleLaunchChildConversationAction,
  type LaunchChildConversationResult,
} from "#/services/child-conversation-launch";
import { ConversationFirstMessageError } from "#/api/conversation-service/agent-server-conversation-service.api";

const {
  mockCreateConversation,
  mockResolveWorkingDir,
  mockSendMessage,
  mockUpdateTitle,
  mockCreateCloudAppConversation,
  mockGetCloudStartTask,
  mockPickCloudBackend,
  mockGetCachedAgentServerVersion,
  mockGetSettings,
  mockListMetaProfiles,
  mockBatchGetAppConversations,
  mockListAgentProfiles,
  mockListLlmProfiles,
  mockSupportsLaunchLlmOverride,
} = vi.hoisted(() => ({
  mockCreateConversation: vi.fn(),
  mockResolveWorkingDir: vi.fn(),
  mockSendMessage: vi.fn(),
  mockUpdateTitle: vi.fn(),
  mockCreateCloudAppConversation: vi.fn(),
  mockGetCloudStartTask: vi.fn(),
  mockPickCloudBackend: vi.fn(),
  mockGetCachedAgentServerVersion: vi.fn(),
  mockGetSettings: vi.fn(),
  mockListMetaProfiles: vi.fn(),
  mockBatchGetAppConversations: vi.fn(),
  mockListAgentProfiles: vi.fn(),
  mockListLlmProfiles: vi.fn(),
  mockSupportsLaunchLlmOverride: vi.fn(),
}));

vi.mock(
  "#/hooks/use-migrate-default-agent-profile",
  async (importOriginal) => ({
    ...(await importOriginal<
      typeof import("#/hooks/use-migrate-default-agent-profile")
    >()),
    ensureDefaultAgentProfileMigrated: vi.fn().mockResolvedValue(true),
  }),
);

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  async (importOriginal) => ({
    // Keep the real named exports (e.g. `ConversationFirstMessageError`, which
    // `resolve-conversation-launch` imports as a value for `instanceof`) and
    // override only the default service object the tests stub.
    ...(await importOriginal<
      typeof import("#/api/conversation-service/agent-server-conversation-service.api")
    >()),
    default: {
      createConversation: mockCreateConversation,
      resolveConversationWorkingDir: mockResolveWorkingDir,
      sendMessage: mockSendMessage,
      updateConversationTitle: mockUpdateTitle,
      batchGetAppConversations: mockBatchGetAppConversations,
    },
  }),
);

vi.mock(
  "#/api/agent-profiles-service/agent-profiles-service.api",
  async (importOriginal) => ({
    ...(await importOriginal<
      typeof import("#/api/agent-profiles-service/agent-profiles-service.api")
    >()),
    default: { listProfiles: mockListAgentProfiles, getProfile: vi.fn() },
  }),
);

vi.mock("#/api/profiles-service/profiles-service.api", () => ({
  default: { listProfiles: mockListLlmProfiles },
}));

vi.mock("#/api/agent-server-adapter", async (importOriginal) => ({
  ...(await importOriginal<typeof import("#/api/agent-server-adapter")>()),
  backendSupportsLaunchLlmOverride: mockSupportsLaunchLlmOverride,
}));

vi.mock("#/api/cloud/conversation-service.api", () => ({
  createCloudAppConversation: mockCreateCloudAppConversation,
  getCloudAppConversationStartTask: mockGetCloudStartTask,
  pickCloudBackendForLaunch: mockPickCloudBackend,
}));

vi.mock("#/api/agent-server-compatibility", () => ({
  getCachedAgentServerVersion: mockGetCachedAgentServerVersion,
  compareAgentServerVersions: (actual: string, required: string) => {
    const parse = (v: string) => v.split(".").map(Number);
    const [a, b] = [parse(actual), parse(required)];
    for (let i = 0; i < 3; i += 1) {
      if (a[i] > b[i]) return 1;
      if (a[i] < b[i]) return -1;
    }
    return 0;
  },
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
  displaySuccessToastWithLink: vi.fn(),
}));

vi.mock("#/api/settings-service/settings-service.api", () => ({
  default: { getSettings: mockGetSettings },
}));

vi.mock("#/api/meta-profiles-service/meta-profiles-service.api", () => ({
  default: { listMetaProfiles: mockListMetaProfiles },
}));

const PARENT_ID = "parent-conversation-id";

function action(
  overrides: Partial<LaunchChildConversationAction>,
): LaunchChildConversationAction {
  return {
    kind: LAUNCH_CHILD_CONVERSATION_ACTION_KIND,
    target: "local",
    task: "Add a regression test for the parser",
    ...overrides,
  } as LaunchChildConversationAction;
}

/** Read back the JSON payload the service posted into the parent conversation. */
function reportedResult(): LaunchChildConversationResult {
  const [, message] = mockSendMessage.mock.calls.at(-1) as [
    string,
    { content: { text: string }[] },
  ];
  return JSON.parse(
    message.content[0].text.slice(CHILD_CONVERSATION_RESULT_PREFIX.length),
  );
}

let toolCall = 0;
const nextToolCallId = () => {
  toolCall += 1;
  return `tool-call-${toolCall}`;
};

/** The repo fields the parent's metadata hands down to its children. */
const PARENT_REPO_METADATA = {
  selected_repository: "octocat/hello-world",
  selected_branch: "main",
  git_provider: null,
};

/**
 * Give the parent a repository/workspace, which is what marks its workspace as
 * able to host a worktree. Without stored metadata the parent is a scratch
 * directory and the worktree is skipped.
 */
function seedWorktreeCapableParent() {
  setStoredConversationMetadata(PARENT_ID, {
    ...PARENT_REPO_METADATA,
    selected_workspace: "/Users/jane/projects/foo",
  });
}

describe("handleLaunchChildConversationAction", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useGoalStore.setState({ statusByConversation: {} });
    vi.clearAllMocks();
    mockResolveWorkingDir.mockResolvedValue("/Users/jane/projects/foo");
    mockCreateConversation.mockResolvedValue({
      id: "start-task-id",
      app_conversation_id: "child-id",
      status: "READY",
    });
    mockUpdateTitle.mockResolvedValue(undefined);
    mockSendMessage.mockResolvedValue(undefined);
    mockGetCachedAgentServerVersion.mockReturnValue("1.37.1");
    mockGetSettings.mockResolvedValue({
      run_router_at_conversation_start: false,
    });
    mockListMetaProfiles.mockResolvedValue({
      meta_profiles: [],
      active_meta_profile: null,
    });
    mockBatchGetAppConversations.mockResolvedValue([]);
    getQueryClient().clear();
    // A parent without a recorded profile launches its child from the active
    // one, the seeded `default` here.
    mockListAgentProfiles.mockResolvedValue({
      profiles: [
        {
          id: "profile-default",
          name: "default",
          agent_kind: "openhands",
          revision: 1,
          llm_profile_ref: "default",
          mcp_server_refs: null,
        },
      ],
      active_agent_profile_id: "profile-default",
    });
    mockListLlmProfiles.mockResolvedValue({
      profiles: [],
      active_profile: null,
    });
    mockSupportsLaunchLlmOverride.mockResolvedValue(true);
  });

  describe("parameter validation", () => {
    // `enum` is advertised to the LLM but dropped when the agent-server builds
    // the action model, so these values arrive intact and must be caught here.
    it.each([
      [
        "an unknown target",
        { target: "clould" },
        '`target` must be exactly "local" or "cloud"',
      ],
      [
        "an unknown isolation",
        { isolation: "worktre" },
        '`isolation` must be exactly "worktree" or "shared"',
      ],
      [
        "a repository on a local target",
        { target: "local", repository: "octocat/hello-world" },
        "A local child always runs in this conversation's workspace",
      ],
      [
        "an isolation on a cloud target",
        { target: "cloud", isolation: "worktree" },
        "Cloud children always run in their own isolated sandbox",
      ],
      [
        "a branch without a repository",
        { target: "cloud", branch: "main" },
        'Pass `repository` as "owner/repo" alongside `branch`',
      ],
      [
        "an empty task",
        { task: "   " },
        "`task` must be a self-contained brief",
      ],
    ])(
      "rejects %s with corrective guidance and launches nothing",
      async (_label, overrides, guidance) => {
        await handleLaunchChildConversationAction(
          action(overrides),
          PARENT_ID,
          nextToolCallId(),
        );

        const result = reportedResult();
        expect(result.status).toBe("error");
        expect(result.status === "error" && result.guidance).toContain(
          guidance,
        );
        expect(mockCreateConversation).not.toHaveBeenCalled();
        expect(mockCreateCloudAppConversation).not.toHaveBeenCalled();
      },
    );
  });

  describe("local target", () => {
    // The agent server rejects a parent in a different workspace, so the child
    // has to request the parent's own directory and isolate via the worktree.
    it("launches into the parent's workspace as a linked, isolated child", async () => {
      seedWorktreeCapableParent();

      await handleLaunchChildConversationAction(
        action({}),
        PARENT_ID,
        nextToolCallId(),
      );

      expect(mockCreateConversation).toHaveBeenCalledWith({
        agentProfileId: "profile-default",
        agentProfileKind: "openhands",
        agentLlmProfileRef: "default",
        initialUserMsg: "Add a regression test for the parser",
        metadata: PARENT_REPO_METADATA,
        workingDirOverride: "/Users/jane/projects/foo",
        workspaceMode: "new_worktree",
        parentConversationId: PARENT_ID,
      });
    });

    it("launches the child from the parent's agent profile and current LLM", async () => {
      seedWorktreeCapableParent();
      mockBatchGetAppConversations.mockResolvedValue([
        {
          agent_kind: "openhands",
          active_profile: "fast",
          launched_agent_profile: {
            agent_profile_id: "profile-1",
            revision: 3,
          },
        },
      ]);
      mockListAgentProfiles.mockResolvedValue({
        profiles: [
          {
            id: "profile-1",
            name: "reviewer",
            agent_kind: "openhands",
            revision: 3,
            llm_profile_ref: "slow",
            mcp_server_refs: null,
          },
        ],
        active_agent_profile_id: null,
      });
      mockListLlmProfiles.mockResolvedValue({
        profiles: [{ name: "slow" }, { name: "fast" }],
        active_profile: "slow",
      });

      await handleLaunchChildConversationAction(
        action({}),
        PARENT_ID,
        nextToolCallId(),
      );

      expect(mockCreateConversation).toHaveBeenCalledWith(
        expect.objectContaining({
          agentProfileId: "profile-1",
          agentProfileKind: "openhands",
          agentLlmProfileOverride: "fast",
          parentConversationId: PARENT_ID,
        }),
      );
      expect(getStoredConversationMetadata("child-id")?.active_profile).toBe(
        "fast",
      );
    });

    it("reports the child's id, url and status back to the agent", async () => {
      seedWorktreeCapableParent();

      await handleLaunchChildConversationAction(
        action({}),
        PARENT_ID,
        nextToolCallId(),
      );

      expect(reportedResult()).toMatchObject({
        status: "launched",
        target: "local",
        conversation_id: "child-id",
        url: "http://localhost:3000/conversations/child-id",
        initial_status: "READY",
        workspace: "/Users/jane/projects/foo",
        isolation: "worktree",
        parent_link: true,
      });
    });

    // Local start requests carry no title field — only `autotitle` — so an
    // explicit title has to be applied as a follow-up rename.
    it("renames the child when a title is given", async () => {
      await handleLaunchChildConversationAction(
        action({ title: "Parser tests" }),
        PARENT_ID,
        nextToolCallId(),
      );

      expect(mockUpdateTitle).toHaveBeenCalledWith("child-id", "Parser tests");
    });

    it("runs a shared-isolation child in the parent's directory itself", async () => {
      await handleLaunchChildConversationAction(
        action({ isolation: "shared" }),
        PARENT_ID,
        nextToolCallId(),
      );

      expect(mockCreateConversation).toHaveBeenCalledWith({
        agentProfileId: "profile-default",
        agentProfileKind: "openhands",
        agentLlmProfileRef: "default",
        initialUserMsg: expect.anything(),
        metadata: null,
        workingDirOverride: "/Users/jane/projects/foo",
        workspaceMode: "local_repo",
        parentConversationId: PARENT_ID,
      });
    });

    // `parent_conversation_id` landed in agent-server 1.37.1; older servers
    // drop it silently, so the agent must not be told the link exists.
    it("reports that an older agent server did not persist the parent link", async () => {
      mockGetCachedAgentServerVersion.mockReturnValue("1.37.0");

      await handleLaunchChildConversationAction(
        action({}),
        PARENT_ID,
        nextToolCallId(),
      );

      const result = reportedResult();
      expect(result).toMatchObject({ status: "launched", parent_link: false });
      expect(result.status === "launched" && result.parent_link_note).toContain(
        "1.37.1",
      );
    });

    it("turns a failed launch into corrective guidance", async () => {
      mockCreateConversation.mockRejectedValue(new Error("boom"));

      await handleLaunchChildConversationAction(
        action({}),
        PARENT_ID,
        nextToolCallId(),
      );

      expect(reportedResult()).toMatchObject({
        status: "error",
        error: "boom",
      });
    });

    // A conversation started without a repository runs in a scratch directory
    // that is `git init`-ed but never committed to. `git worktree add` cannot
    // branch from an unborn HEAD, and the agent-server raises that as a 500
    // that used to take the whole launch down.
    it("skips the worktree when the parent workspace cannot host one", async () => {
      await handleLaunchChildConversationAction(
        action({}),
        PARENT_ID,
        nextToolCallId(),
      );

      expect(mockCreateConversation).toHaveBeenCalledTimes(1);
      expect(mockCreateConversation).toHaveBeenCalledWith({
        agentProfileId: "profile-default",
        agentProfileKind: "openhands",
        agentLlmProfileRef: "default",
        initialUserMsg: expect.anything(),
        metadata: null,
        workingDirOverride: "/Users/jane/projects/foo",
        workspaceMode: "local_repo",
        parentConversationId: PARENT_ID,
      });

      const result = reportedResult();
      expect(result).toMatchObject({ status: "launched", isolation: "shared" });
      expect(result.status === "launched" && result.isolation_note).toContain(
        "no commits",
      );
    });

    // The metadata check cannot see the workspace's git state, so a worktree
    // that fails anyway must not lose the launch.
    it("falls back to a shared child when the worktree cannot be created", async () => {
      seedWorktreeCapableParent();
      mockCreateConversation.mockRejectedValueOnce(
        new Error("fatal: not a valid object name: 'HEAD'"),
      );

      await handleLaunchChildConversationAction(
        action({}),
        PARENT_ID,
        nextToolCallId(),
      );

      expect(mockCreateConversation).toHaveBeenCalledTimes(2);
      expect(mockCreateConversation).toHaveBeenLastCalledWith({
        agentProfileId: "profile-default",
        agentProfileKind: "openhands",
        agentLlmProfileRef: "default",
        initialUserMsg: expect.anything(),
        metadata: PARENT_REPO_METADATA,
        workingDirOverride: "/Users/jane/projects/foo",
        workspaceMode: "local_repo",
        parentConversationId: PARENT_ID,
      });

      const result = reportedResult();
      expect(result).toMatchObject({
        status: "launched",
        conversation_id: "child-id",
        isolation: "shared",
      });
      // The agent promised the user an isolated child, so it has to learn that
      // this one shares the parent's directory after all.
      expect(result.status === "launched" && result.isolation_note).toContain(
        "not a valid object name",
      );
    });

    // Both attempts failing means the worktree was not the problem, so the
    // agent gets the original error rather than the fallback's.
    it("reports the original failure when the shared fallback also fails", async () => {
      seedWorktreeCapableParent();
      mockCreateConversation
        .mockRejectedValueOnce(new Error("worktree boom"))
        .mockRejectedValueOnce(new Error("fallback boom"));

      await handleLaunchChildConversationAction(
        action({}),
        PARENT_ID,
        nextToolCallId(),
      );

      expect(mockCreateConversation).toHaveBeenCalledTimes(2);
      expect(reportedResult()).toMatchObject({
        status: "error",
        error: "worktree boom",
      });
    });

    it("does not retry a child that never asked for a worktree", async () => {
      mockCreateConversation.mockRejectedValue(new Error("boom"));

      await handleLaunchChildConversationAction(
        action({ isolation: "shared" }),
        PARENT_ID,
        nextToolCallId(),
      );

      expect(mockCreateConversation).toHaveBeenCalledTimes(1);
      expect(reportedResult()).toMatchObject({
        status: "error",
        error: "boom",
      });
    });

    // The withheld-first-message failure means the conversation exists and only
    // its first turn did not start. The worktree fallback must not treat it as
    // a worktree failure: retrying with `shared` would create a second
    // conversation and downgrade the isolation the caller asked for.
    it("recovers onto the created child when the first message fails, without downgrading isolation", async () => {
      seedWorktreeCapableParent();
      mockCreateConversation.mockRejectedValueOnce(
        new ConversationFirstMessageError(
          "child-id",
          [{ type: "text", text: "Add a regression test for the parser" }],
          new Error("network timeout"),
        ),
      );

      await handleLaunchChildConversationAction(
        action({}),
        PARENT_ID,
        nextToolCallId(),
      );

      // One attempt only — no second conversation, no `shared` retry.
      expect(mockCreateConversation).toHaveBeenCalledTimes(1);
      expect(mockCreateConversation).toHaveBeenCalledWith(
        expect.objectContaining({ workspaceMode: "new_worktree" }),
      );

      const result = reportedResult();
      expect(result).toMatchObject({
        status: "launched",
        conversation_id: "child-id",
        isolation: "worktree",
      });
      // The isolation was not downgraded, so no fallback note is attached.
      expect(
        result.status === "launched" && result.isolation_note,
      ).toBeUndefined();
      // An ambiguous failure (transport error) may already have been accepted,
      // so the note must not tell the agent the child is idle.
      const note =
        result.status === "launched" ? result.first_message_note : undefined;
      expect(note).toContain("network timeout");
      expect(note).toContain("check its events");
      expect(note).toContain("Do not re-run the delegation");
    });

    // A 4xx is definitive: the send was refused, so the child is genuinely idle
    // and re-sending cannot duplicate it.
    it("tells the agent a definitively rejected child is idle", async () => {
      seedWorktreeCapableParent();
      const rejection = new Error("unprocessable") as Error & {
        name: string;
        status: number;
      };
      rejection.name = "HttpError";
      rejection.status = 422;
      mockCreateConversation.mockRejectedValueOnce(
        new ConversationFirstMessageError(
          "child-id",
          [{ type: "text", text: "Add a regression test for the parser" }],
          rejection,
        ),
      );

      await handleLaunchChildConversationAction(
        action({}),
        PARENT_ID,
        nextToolCallId(),
      );

      const result = reportedResult();
      const note =
        result.status === "launched" ? result.first_message_note : undefined;
      expect(note).toContain("was refused");
      expect(note).toContain("It is idle");
    });

    // The recovery must not silently drop the caller's title or a computed
    // isolation note (a scratch workspace falls back to `shared` before the
    // send, and that downgrade has to be reported).
    it("keeps the requested title and the pre-computed isolation note on recovery", async () => {
      // No stored metadata → scratch workspace → shared fallback note exists.
      mockCreateConversation.mockRejectedValueOnce(
        new ConversationFirstMessageError(
          "child-id",
          [{ type: "text", text: "Add a regression test for the parser" }],
          new Error("network timeout"),
        ),
      );

      await handleLaunchChildConversationAction(
        action({ title: "Parser tests" }),
        PARENT_ID,
        nextToolCallId(),
      );

      expect(mockUpdateTitle).toHaveBeenCalledWith("child-id", "Parser tests");
      const result = reportedResult();
      expect(result).toMatchObject({
        status: "launched",
        conversation_id: "child-id",
        title: "Parser tests",
        isolation: "shared",
      });
      expect(result.status === "launched" && result.isolation_note).toContain(
        "no commits",
      );
    });

    // The same recovery applies when no worktree was involved: a shared child
    // whose first message failed is recovered, not reported as a failed launch.
    it("recovers a shared child whose first message failed", async () => {
      mockCreateConversation.mockRejectedValueOnce(
        new ConversationFirstMessageError(
          "child-id",
          [{ type: "text", text: "Add a regression test for the parser" }],
          new Error("network timeout"),
        ),
      );

      await handleLaunchChildConversationAction(
        action({ isolation: "shared" }),
        PARENT_ID,
        nextToolCallId(),
      );

      expect(mockCreateConversation).toHaveBeenCalledTimes(1);
      expect(reportedResult()).toMatchObject({
        status: "launched",
        conversation_id: "child-id",
        isolation: "shared",
      });
    });
  });

  describe("cloud target", () => {
    const cloudBackend = {
      id: "cloud-1",
      kind: "cloud" as const,
      host: "https://app.all-hands.dev",
      apiKey: "secret",
      name: "Cloud",
    };

    beforeEach(() => {
      mockPickCloudBackend.mockReturnValue(cloudBackend);
    });

    it("starts the child on the connected cloud backend without a dangling parent link", async () => {
      mockCreateCloudAppConversation.mockResolvedValue({
        id: "start-task-id",
        app_conversation_id: "cloud-child-id",
        status: "READY",
      });

      await handleLaunchChildConversationAction(
        action({
          target: "cloud",
          repository: "octocat/hello-world",
          branch: "main",
        }),
        PARENT_ID,
        nextToolCallId(),
      );

      expect(mockCreateCloudAppConversation).toHaveBeenCalledWith(
        expect.objectContaining({
          selected_repository: "octocat/hello-world",
          selected_branch: "main",
          // The parent lives on the local agent server, so Cloud has no
          // conversation to link the child to — and a dangling id would hide
          // the child from the Cloud conversation list.
          parent_conversation_id: null,
        }),
        cloudBackend,
      );
    });

    it("skips the meta-profiles fetch and launch additions when the router toggle is off", async () => {
      mockCreateCloudAppConversation.mockResolvedValue({
        id: "start-task-id",
        app_conversation_id: "cloud-child-id",
        status: "READY",
      });

      await handleLaunchChildConversationAction(
        action({ target: "cloud" }),
        PARENT_ID,
        nextToolCallId(),
      );

      expect(mockListMetaProfiles).not.toHaveBeenCalled();
      const [request] = mockCreateCloudAppConversation.mock.calls.at(-1) as [
        Record<string, unknown>,
      ];
      expect(request).not.toHaveProperty("agent_launch_additions");
    });

    it("stamps the router instruction when the toggle is on and a meta-profile is active", async () => {
      mockGetSettings.mockResolvedValue({
        run_router_at_conversation_start: true,
      });
      mockListMetaProfiles.mockResolvedValue({
        meta_profiles: [{ name: "router" }],
        active_meta_profile: "router",
      });
      mockCreateCloudAppConversation.mockResolvedValue({
        id: "start-task-id",
        app_conversation_id: "cloud-child-id",
        status: "READY",
      });

      await handleLaunchChildConversationAction(
        action({ target: "cloud" }),
        PARENT_ID,
        nextToolCallId(),
      );

      expect(mockCreateCloudAppConversation).toHaveBeenCalledWith(
        expect.objectContaining({
          agent_launch_additions: {
            system_message_suffix_append: expect.stringContaining(
              "route_task_to_model",
            ),
          },
        }),
        cloudBackend,
      );
    });

    // Cloud provisions the sandbox asynchronously and only fills in
    // `app_conversation_id` at READY, so reporting the first response would
    // hand the agent a result with no conversation to open.
    it("waits for the sandbox to expose the conversation id before reporting", async () => {
      mockCreateCloudAppConversation.mockResolvedValue({
        id: "start-task-id",
        app_conversation_id: null,
        status: "WORKING",
      });
      mockGetCloudStartTask.mockResolvedValue({
        id: "start-task-id",
        app_conversation_id: "cloud-child-id",
        status: "READY",
      });

      await handleLaunchChildConversationAction(
        action({ target: "cloud" }),
        PARENT_ID,
        nextToolCallId(),
      );

      expect(reportedResult()).toMatchObject({
        status: "launched",
        target: "cloud",
        conversation_id: "cloud-child-id",
        url: "https://app.all-hands.dev/canvas/conversations/cloud-child-id",
        initial_status: "READY",
        parent_link: false,
      });
    });

    // Cloud and self-hosted Enterprise serve the legacy UI at the host root and
    // Agent Canvas under /canvas.
    it("links the child to the backend's Agent Canvas, not its legacy UI", async () => {
      mockPickCloudBackend.mockReturnValueOnce({
        ...cloudBackend,
        host: "https://openhands.example.com/",
      });
      mockCreateCloudAppConversation.mockResolvedValue({
        id: "start-task-id",
        app_conversation_id: "cloud-child-id",
        status: "READY",
      });

      await handleLaunchChildConversationAction(
        action({ target: "cloud" }),
        PARENT_ID,
        nextToolCallId(),
      );

      expect(reportedResult()).toMatchObject({
        url: "https://openhands.example.com/canvas/conversations/cloud-child-id",
      });
    });

    it("guides the agent back to local when no cloud backend is connected", async () => {
      mockPickCloudBackend.mockReturnValueOnce(null);

      await handleLaunchChildConversationAction(
        action({ target: "cloud" }),
        PARENT_ID,
        nextToolCallId(),
      );

      const result = reportedResult();
      expect(result.status).toBe("error");
      expect(result.status === "error" && result.guidance).toContain(
        'target="local"',
      );
      expect(mockCreateCloudAppConversation).not.toHaveBeenCalled();
    });
  });

  // A replayed ActionEvent (socket reconnect, or a REST/WebSocket race after a
  // reload) must not start a second — on Cloud, billable — conversation.
  it("ignores a replayed tool call", async () => {
    const toolCallId = nextToolCallId();

    await handleLaunchChildConversationAction(
      action({}),
      PARENT_ID,
      toolCallId,
    );
    await handleLaunchChildConversationAction(
      action({}),
      PARENT_ID,
      toolCallId,
    );

    expect(mockCreateConversation).toHaveBeenCalledTimes(1);
    expect(mockSendMessage).toHaveBeenCalledTimes(1);
  });

  // The agent server cancels an active /goal loop on any inbound message, so
  // the result stays in the toast rather than ending the user's loop.
  it("does not post the result while a goal loop is running", async () => {
    useGoalStore.setState({
      statusByConversation: {
        [PARENT_ID]: {
          active: true,
          status: "running",
          iteration: 1,
          max_iterations: 5,
          objective: "ship it",
          verdict: null,
        },
      },
    });

    await handleLaunchChildConversationAction(
      action({}),
      PARENT_ID,
      nextToolCallId(),
    );

    expect(mockCreateConversation).toHaveBeenCalledTimes(1);
    expect(mockSendMessage).not.toHaveBeenCalled();
  });
});

describe("isLaunchChildConversationActionEvent", () => {
  const makeActionEvent = (toolName: string) =>
    ({
      id: "evt-1",
      timestamp: "2026-07-26T00:00:00Z",
      source: "agent",
      action: {
        kind: LAUNCH_CHILD_CONVERSATION_ACTION_KIND,
        target: "local",
        task: "do the thing",
      },
      tool_name: toolName,
      tool_call_id: "call-1",
    }) as never;

  it("returns true for an ActionEvent from the launch tool", () => {
    expect(
      isLaunchChildConversationActionEvent(
        makeActionEvent(LAUNCH_CHILD_CONVERSATION_TOOL_NAME),
      ),
    ).toBe(true);
  });

  it("returns false when tool_name belongs to a different tool", () => {
    expect(
      isLaunchChildConversationActionEvent(
        makeActionEvent("canvas_ui_control"),
      ),
    ).toBe(false);
  });
});
