import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { DebugAutomationButton } from "#/components/features/automations/detail/debug-automation-button";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
} from "#/types/automation";

// Mock the underlying services (not the useCreateConversation hook), so the
// real hook logic — id mapping + onSuccess navigation — is exercised.
vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: { createConversation: vi.fn() },
  }),
);

vi.mock("#/api/profiles-service/profiles-service.api", () => ({
  default: { listProfiles: vi.fn() },
}));

type CreatedConversation = Awaited<
  ReturnType<typeof AgentServerConversationService.createConversation>
>;

const localBackend: Backend = {
  id: "local-1",
  name: "Local",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};

const failedRun: AutomationRun = {
  id: "run-9",
  status: AutomationRunStatus.FAILED,
  conversation_id: "conv-1",
  bash_command_id: "cmd-1",
  error_detail: "Process exited with code 1",
  started_at: "2026-01-01T10:00:00Z",
  completed_at: "2026-01-01T10:02:00Z",
};

const automation: Automation = {
  id: "auto-1",
  name: "Daily Jira digest",
  prompt: "Summarize new Jira issues",
  trigger: { type: "cron", schedule: "0 9 * * *" },
  enabled: true,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

function renderButton(stderr = "HTTP Error 410: Gone") {
  const navigate = vi.fn();
  const navValue: NavigationContextValue = {
    currentPath: "/automations/auto-1",
    conversationId: null,
    isNavigating: false,
    navigate,
  };
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <NavigationProvider value={navValue}>
          <DebugAutomationButton
            run={failedRun}
            automation={automation}
            stderr={stderr}
          />
        </NavigationProvider>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
  return { navigate };
}

beforeEach(() => {
  vi.clearAllMocks();
  __resetActiveStoreForTests();
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
  vi.mocked(ProfilesService.listProfiles).mockResolvedValue({
    profiles: [],
    active_profile: null,
  });
});

afterEach(() => {
  __resetActiveStoreForTests();
});

describe("DebugAutomationButton", () => {
  it("seeds a new conversation with the run's error details and navigates to it", async () => {
    // Arrange — a null app_conversation_id resolves to a `task-<id>` route.
    vi.mocked(
      AgentServerConversationService.createConversation,
    ).mockResolvedValue({
      id: "abc123",
      app_conversation_id: null,
      agent_server_url: null,
    } as CreatedConversation);
    const { navigate } = renderButton("HTTP Error 410: Gone");

    // Act
    fireEvent.click(screen.getByTestId("debug-automation-button"));

    // Assert — the first arg is the initial user message (the debug prompt).
    await waitFor(() =>
      expect(
        AgentServerConversationService.createConversation,
      ).toHaveBeenCalledTimes(1),
    );
    const query = vi.mocked(AgentServerConversationService.createConversation)
      .mock.calls[0][0];
    expect(query).toContain("HTTP Error 410: Gone");
    expect(query).toContain("Daily Jira digest");

    await waitFor(() =>
      expect(navigate).toHaveBeenCalledWith("/conversations/task-abc123"),
    );
  });

  it("disables the button while the conversation is being created", async () => {
    // Arrange — a never-settling promise keeps the mutation pending.
    vi.mocked(
      AgentServerConversationService.createConversation,
    ).mockReturnValue(new Promise<CreatedConversation>(() => {}));
    renderButton();

    const button = screen.getByTestId("debug-automation-button");

    // Act
    fireEvent.click(button);

    // Assert
    await waitFor(() => expect(button).toBeDisabled());
    expect(button).toHaveAttribute("aria-busy", "true");
  });
});
