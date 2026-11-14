import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../test-utils";
import { ResourcesPanel } from "#/components/features/conversation/resources-panel";
import ConversationResourcesService, {
  type ConversationResources,
} from "#/api/conversation-resources-service";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  getActiveBackend,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import {
  getConversationState,
  setConversationState,
} from "#/utils/conversation-local-storage";

const load = vi.spyOn(ConversationResourcesService, "load");
const getConversations = vi.spyOn(
  AgentServerConversationService,
  "batchGetAppConversations",
);

function resources(): ConversationResources {
  return {
    agentKind: "openhands",
    model: "actual-runtime-model",
    profile: {
      id: "profile-id",
      name: "Code review",
      launchedRevision: 3,
      currentRevision: 3,
    },
    skills: [
      {
        name: "project-guide",
        type: "knowledge",
        source: "/workspace/.agents/skills/project-guide/SKILL.md",
      },
      { name: "public-guide", type: "knowledge", source: "public" },
    ],
    mcpServers: [{ name: "docs", healthKey: "docs-health", type: "stdio" }],
    secretNames: ["GITHUB_TOKEN"],
    secretScopeOnly: false,
    condenser: { kind: "LLMSummarizingCondenser", maxSize: 240, keepFirst: 2 },
    verification: { kind: null },
  };
}

function appConversation(id = "test-conversation-id"): AppConversation {
  return {
    id,
    created_by_user_id: null,
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    title: "Resources test",
    trigger: null,
    pr_number: [],
    llm_model: "app-list-model",
    metrics: null,
    created_at: "2026-11-01T00:00:00Z",
    updated_at: "2026-11-01T00:00:00Z",
    execution_status: null,
    conversation_url: null,
    session_api_key: null,
    sandbox_id: null,
    sub_conversation_ids: [],
  };
}

function renderPanel() {
  return renderWithProviders(
    <ActiveBackendProvider>
      <ResourcesPanel />
    </ActiveBackendProvider>,
  );
}

describe("ResourcesPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setRegisteredBackends([
      {
        id: "resources-local",
        name: "Local",
        kind: "local",
        host: "http://local.test",
        apiKey: "",
      },
    ]);
    setActiveSelection({ backendId: "resources-local" });
    getConversations.mockResolvedValue([appConversation()]);
    load.mockResolvedValue(resources());
  });

  it("expands effective resources and links each category to its settings owner", async () => {
    const user = userEvent.setup();
    renderPanel();
    await screen.findByText("actual-runtime-model");
    expect(screen.queryByText("app-list-model")).not.toBeInTheDocument();
    await user.click(screen.getByText("RESOURCES$PROFILE"));
    expect(screen.getByText("Code review")).toBeVisible();
    expect(screen.getByText("RESOURCES$REVISION")).toBeVisible();
    expect(
      screen.queryByText("RESOURCES$PROFILE_CHANGED"),
    ).not.toBeInTheDocument();
    await user.click(screen.getByText("SETTINGS$SKILLS_TITLE"));
    expect(screen.getByText("project-guide")).toBeVisible();
    expect(screen.getByText("public-guide")).toBeVisible();
    const links = screen
      .getAllByRole("link", { hidden: true })
      .map((link) => link.getAttribute("href"));
    expect(links).toEqual([
      "/settings/agents",
      "/settings/llm",
      "/skills",
      "/mcp",
      "/settings/secrets",
      "/settings/agents",
      "/settings/verification",
    ]);
  });

  it("warns when the current profile revision differs from the launch stamp", async () => {
    const data = resources();
    data.profile!.currentRevision = 4;
    load.mockResolvedValue(data);
    renderPanel();
    expect(
      await screen.findByText("RESOURCES$PROFILE_CHANGED"),
    ).toBeInTheDocument();
  });

  it("makes missing launch provenance explicit while retaining the reported model", async () => {
    load.mockResolvedValue({ ...resources(), profile: null });
    renderPanel();
    expect(await screen.findByText("RESOURCES$NO_PROFILE")).toBeInTheDocument();
    expect(screen.getByText("actual-runtime-model")).toBeInTheDocument();
  });

  it("shows ACP-reported model and MCP, with OpenHands-only categories unavailable", async () => {
    load.mockResolvedValue({
      ...resources(),
      agentKind: "acp",
      model: "acp-reported-model",
    });
    renderPanel();
    expect(await screen.findByText("acp-reported-model")).toBeInTheDocument();
    expect(screen.getByText("docs")).toBeInTheDocument();
    expect(screen.queryByText("project-guide")).not.toBeInTheDocument();
    expect(
      screen.queryByText("LLMSummarizingCondenser"),
    ).not.toBeInTheDocument();
    expect(screen.getAllByText("RESOURCES$UNAVAILABLE")).toHaveLength(3);
  });

  it("uses Cloud runtime context and hides profile fields Cloud does not apply", async () => {
    const local = getActiveBackend().backend;
    setRegisteredBackends([{ ...local, kind: "cloud" }]);
    setActiveSelection({ backendId: local.id, orgId: "org-id" });
    getConversations.mockResolvedValue([
      {
        ...appConversation(),
        conversation_url:
          "https://sandbox.test/api/conversations/test-conversation-id",
        session_api_key: "sandbox-key",
      },
    ]);
    renderPanel();
    await screen.findByText("actual-runtime-model");
    expect(load).toHaveBeenCalledWith(
      "test-conversation-id",
      expect.objectContaining({ orgId: "org-id" }),
      {
        conversationUrl:
          "https://sandbox.test/api/conversations/test-conversation-id",
        sessionApiKey: "sandbox-key",
      },
    );
    expect(screen.queryByText("project-guide")).not.toBeInTheDocument();
    expect(screen.queryByText("GITHUB_TOKEN")).not.toBeInTheDocument();
    expect(screen.getAllByText("RESOURCES$UNAVAILABLE")).toHaveLength(2);
  });

  it("renders names only and keeps unknown data distinct from empty lists", async () => {
    load.mockResolvedValue({
      ...resources(),
      skills: null,
      mcpServers: [],
      secretNames: ["GITHUB_TOKEN"],
    });
    renderPanel();
    const panel = await screen.findByTestId("resources-panel");
    expect(within(panel).getByText("GITHUB_TOKEN")).toBeInTheDocument();
    expect(
      within(panel).getByText("RESOURCES$UNAVAILABLE"),
    ).toBeInTheDocument();
    expect(
      within(panel).getByText("CONVERSATION$OVERVIEW_NONE"),
    ).toBeInTheDocument();
  });

  it("retries a failed runtime read without starting or resuming the conversation", async () => {
    load
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce(resources());
    const user = userEvent.setup();
    renderPanel();
    await user.click(
      await screen.findByRole("button", { name: "SETTINGS$APPS_VIEW_RETRY" }),
    );
    expect(await screen.findByText("actual-runtime-model")).toBeInTheDocument();
    expect(load).toHaveBeenCalledTimes(2);
    expect(getConversations).toHaveBeenCalledTimes(1);
  });

  it("drops the old panel data on a backend switch even if its runtime request resolves late", async () => {
    let resolve!: (data: ConversationResources) => void;
    load.mockReturnValue(
      new Promise((done) => {
        resolve = done;
      }),
    );
    renderPanel();
    await waitFor(() => expect(load).toHaveBeenCalledTimes(1));
    act(() => {
      setRegisteredBackends([
        {
          id: "other",
          name: "Other",
          kind: "local",
          host: "http://other.test",
          apiKey: "",
        },
      ]);
      setActiveSelection({ backendId: "other" });
      resolve(resources());
    });
    await waitFor(() =>
      expect(
        screen.queryByText("actual-runtime-model"),
      ).not.toBeInTheDocument(),
    );
    expect(load).toHaveBeenCalledTimes(1);
  });

  it("hydrates a Resources selection independently for each conversation", () => {
    setConversationState("first", { selectedTab: "resources" });
    setConversationState("second", { selectedTab: "files" });
    expect(getConversationState("first").selectedTab).toBe("resources");
    expect(getConversationState("second").selectedTab).toBe("files");
  });
});
