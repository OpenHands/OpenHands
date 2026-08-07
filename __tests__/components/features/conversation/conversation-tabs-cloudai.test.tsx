import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderWithProviders } from "test-utils";
import { ConversationTabs } from "#/components/features/conversation/conversation-tabs/conversation-tabs";
import { useConversationAppwriteIntegration } from "#/hooks/query/use-appwrite-integration";
import { appwriteApiKeySecretName } from "#/utils/appwrite-integration-secrets";

vi.mock("#/hooks/query/use-appwrite-integration", () => ({
  useConversationAppwriteIntegration: vi.fn(),
}));

vi.mock("#/hooks/use-task-list", () => ({
  useTaskList: () => ({ hasTaskList: false }),
}));

vi.mock("#/hooks/use-handle-build-plan-click", () => ({
  useHandleBuildPlanClick: () => ({ handleBuildPlanClick: vi.fn() }),
}));

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => ({ curAgentState: "idle" }),
}));

vi.mock("#/contexts/active-backend-context", async () => {
  const actual = await vi.importActual<
    typeof import("#/contexts/active-backend-context")
  >("#/contexts/active-backend-context");
  return {
    ...actual,
    useActiveBackend: () => ({
      backend: {
        kind: "local",
        id: "default-local",
        host: "http://localhost:8000",
      },
    }),
  };
});

const WORKSPACE_ID = "ws-demo";

describe("ConversationTabs CloudAI gate", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("hides CloudAI when the conversation workspace AppWrite is not ready", () => {
    vi.mocked(useConversationAppwriteIntegration).mockReturnValue({
      workspaceId: WORKSPACE_ID,
      config: {
        enabled: false,
        endpoint: "",
        projectId: "",
      },
      apiKeyIsSet: false,
      isReady: false,
      isLoading: false,
      secretName: appwriteApiKeySecretName(WORKSPACE_ID),
    });
    renderWithProviders(<ConversationTabs />);
    expect(
      document.querySelector('[data-aria-label="COMMON$CLOUDAI"]'),
    ).toBeNull();
  });

  it("shows CloudAI when the conversation workspace AppWrite is ready", () => {
    vi.mocked(useConversationAppwriteIntegration).mockReturnValue({
      workspaceId: WORKSPACE_ID,
      config: {
        enabled: true,
        endpoint: "https://cloud.appwrite.io/v1",
        projectId: "proj",
      },
      apiKeyIsSet: true,
      isReady: true,
      isLoading: false,
      secretName: appwriteApiKeySecretName(WORKSPACE_ID),
    });
    renderWithProviders(<ConversationTabs />);
    expect(
      document.querySelector('[data-aria-label="COMMON$CLOUDAI"]'),
    ).toBeTruthy();
  });
});
