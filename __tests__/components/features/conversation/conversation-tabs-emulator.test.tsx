import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderWithProviders } from "test-utils";
import { ConversationTabs } from "#/components/features/conversation/conversation-tabs/conversation-tabs";

vi.mock("#/hooks/query/use-appwrite-integration", () => ({
  useConversationAppwriteIntegration: () => ({
    isReady: false,
    isLoading: false,
  }),
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

const capabilityMock = vi.fn<(cap: string) => boolean>(() => false);

vi.mock("#/hooks/use-pentest-capabilities", () => ({
  useHasPentestCapability: (cap: string) => capabilityMock(cap),
  usePentestCapabilities: () => [],
  usePentestCapabilitiesQuery: () => ({ data: { capabilities: [] } }),
}));

describe("ConversationTabs Emulator", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    capabilityMock.mockReturnValue(false);
  });

  it("hides Emulator tab without pentest.mobile.dynamic (AC-192-2)", () => {
    renderWithProviders(<ConversationTabs />);
    expect(
      document.querySelector('[data-aria-label="COMMON$EMULATOR"]'),
    ).toBeNull();
  });

  it("shows Emulator tab when capability is present (AC-192-2)", () => {
    capabilityMock.mockImplementation(
      (cap: string) => cap === "pentest.mobile.dynamic",
    );
    renderWithProviders(<ConversationTabs />);
    expect(
      document.querySelector('[data-aria-label="COMMON$EMULATOR"]'),
    ).not.toBeNull();
  });
});
