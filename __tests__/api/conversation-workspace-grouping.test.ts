import { describe, expect, it } from "vitest";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { ExecutionStatus } from "#/types/agent-server/core";
import {
  buildConversationMetadataTags,
  resolveSelectedWorkspaceForGrouping,
} from "#/api/conversation-workspace-grouping";

const base: Omit<AppConversation, "id" | "title"> = {
  selected_repository: null,
  selected_branch: null,
  git_provider: null,
  updated_at: "2024-01-02T00:00:00.000Z",
  created_at: "2024-01-01T00:00:00.000Z",
  execution_status: ExecutionStatus.FINISHED,
  conversation_url: null,
  created_by_user_id: null,
  metrics: null,
  llm_model: null,
  trigger: null,
  pr_number: [],
  session_api_key: null,
  sandbox_id: null,
  sub_conversation_ids: [],
};

describe("conversation-workspace-grouping", () => {
  it("buildConversationMetadataTags omits empty workspace paths", () => {
    expect(
      buildConversationMetadataTags({
        selectedWorkspace: "  /tmp/project///  ",
        gitProvider: "github",
      }),
    ).toEqual({
      git_provider: "github",
      workspace: "/tmp/project",
    });
  });

  it("resolveSelectedWorkspaceForGrouping matches registered working_dir for legacy conversations", () => {
    const conversation: AppConversation = {
      ...base,
      id: "legacy",
      title: "legacy",
      selected_workspace: null,
      workspace: { working_dir: "/data/my-app" },
      tags: null,
    };

    expect(
      resolveSelectedWorkspaceForGrouping(conversation, [
        "/other",
        "/data/my-app",
      ]),
    ).toBe("/data/my-app");
  });

  it("resolveSelectedWorkspaceForGrouping does not group arbitrary worktree paths", () => {
    const conversation: AppConversation = {
      ...base,
      id: "worktree",
      title: "worktree",
      selected_workspace: null,
      workspace: { working_dir: "/workspace/project/abc123" },
      tags: null,
    };

    expect(
      resolveSelectedWorkspaceForGrouping(conversation, ["/workspace/project"]),
    ).toBeNull();
  });
});
