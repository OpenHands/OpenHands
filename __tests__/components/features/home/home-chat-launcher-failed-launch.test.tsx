import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClientProvider } from "@tanstack/react-query";
import toast, { Toaster } from "react-hot-toast";
import { HttpError } from "@openhands/typescript-client";

import { HomeChatLauncher } from "#/components/features/home/home-chat-launcher";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import WorkspacesService from "#/api/workspaces-service/workspaces-service.api";
import { HOME_PROMPT_DRAFT_KEY } from "#/hooks/chat/use-draft-persistence";
import { createAgentServerQueryClient } from "#/query-client-config";
import { useConversationStore } from "#/stores/conversation-store";

// The launch failure path is exercised end to end: the real composer, the real
// conversation store, the real toast handlers and the app's query client (with
// its global error toasts). Only the network-facing services are mocked.

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({
    currentPath: "/",
    conversationId: null,
    isNavigating: false,
    navigate: vi.fn(),
  }),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: {
      id: "local-id",
      name: "Local",
      host: "http://localhost",
      apiKey: "test",
      kind: "local" as const,
    },
    orgId: null,
  }),
}));

vi.mock("#/hooks/use-llm-configured", () => ({
  useLlmConfigured: () => ({ isConfigured: true, isLoading: false }),
}));

vi.mock("#/hooks/query/use-conversation-workspace", () => ({
  useConversationWorkspace: () => ({
    isolated: false,
    unsupportedMessage: null,
  }),
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({ trackConversationCreated: vi.fn() }),
}));

// The composer's model/profile controls and the automation sections below it
// issue their own requests; none of them take part in launching.
vi.mock("#/components/features/chat/components/chat-input-actions", () => ({
  ChatInputActions: ({ handleSubmit }: { handleSubmit: () => void }) => (
    <button type="button" data-testid="submit-button" onClick={handleSubmit} />
  ),
}));

vi.mock(
  "#/components/features/automations/recommended-automations-launcher",
  () => ({ RecommendedAutomationsLauncher: () => null }),
);

vi.mock(
  "#/components/features/home/featured-automations/pinned-automations-dashboard",
  () => ({ PinnedAutomationsDashboard: () => null }),
);

vi.mock(
  "#/components/features/home/featured-automations/running-automations-list",
  () => ({ RunningAutomationsList: () => null }),
);

const PROMPT = "Refactor the billing module";
const UNREACHABLE_MESSAGE =
  'HTTP request failed (502 Bad Gateway): "Bad Gateway: connect ECONNREFUSED 127.0.0.1:18811"';

const unreachable = () =>
  new HttpError(
    502,
    "Bad Gateway",
    "Bad Gateway: connect ECONNREFUSED 127.0.0.1:18811",
    UNREACHABLE_MESSAGE,
  );

const renderLauncher = () =>
  render(
    <QueryClientProvider client={createAgentServerQueryClient()}>
      <HomeChatLauncher />
      <Toaster />
    </QueryClientProvider>,
  );

// jsdom has no layout, so `innerText` (which the composer reads on submit) is
// undefined; mirror `textContent` the way a browser would for plain text.
const originalInnerText = Object.getOwnPropertyDescriptor(
  HTMLElement.prototype,
  "innerText",
);

async function submitPrompt() {
  const input = await screen.findByTestId("chat-input");
  input.textContent = PROMPT;
  fireEvent.input(input);
  await userEvent.setup().click(screen.getByTestId("submit-button"));
  return input;
}

describe("HomeChatLauncher failed launch", () => {
  beforeEach(() => {
    Object.defineProperty(HTMLElement.prototype, "innerText", {
      configurable: true,
      get() {
        return this.textContent ?? "";
      },
      set(value: string) {
        this.textContent = value;
      },
    });
    // <Toaster /> asks for the reduced-motion preference, which jsdom lacks.
    vi.stubGlobal("matchMedia", vi.fn().mockReturnValue({ matches: true }));
    vi.spyOn(WorkspacesService, "listWorkspaces").mockResolvedValue({
      workspaces: [],
      workspaceParents: [],
    });
    vi.spyOn(ProfilesService, "listProfiles").mockResolvedValue({
      profiles: [],
      active_profile: null,
    } as never);
    // The seeded OpenHands `default` profile makes the launch path revalidate
    // the profile detail before it posts the conversation.
    vi.spyOn(AgentProfilesService, "listProfiles").mockResolvedValue({
      profiles: [
        { id: "default-id", name: "default", agent_kind: "openhands" },
      ],
      active_agent_profile_id: "default-id",
    } as never);
  });

  afterEach(() => {
    if (originalInnerText) {
      Object.defineProperty(
        HTMLElement.prototype,
        "innerText",
        originalInnerText,
      );
    } else {
      Reflect.deleteProperty(HTMLElement.prototype, "innerText");
    }
    toast.remove();
    sessionStorage.removeItem(HOME_PROMPT_DRAFT_KEY);
    useConversationStore.getState().clearMessageRestoreIfEmpty();
    vi.restoreAllMocks();
  });

  it("shows one error toast, dismisses the loading toast and keeps the prompt when the agent server is unreachable", async () => {
    // Arrange — every launch-path request fails the same way.
    vi.spyOn(AgentProfilesService, "getProfile").mockRejectedValue(
      unreachable(),
    );
    vi.spyOn(
      AgentServerConversationService,
      "createConversation",
    ).mockRejectedValue(unreachable());
    renderLauncher();

    // Act
    const input = await submitPrompt();

    // Assert — the loading toast shows, then gives way to a single error.
    // Error toasts outlive the loading toast's exit, so every one that was
    // raised is still on screen when the loading toast is gone.
    await screen.findByText("HOME$CREATING_CONVERSATION");
    await waitFor(
      () =>
        expect(
          screen.queryByText("HOME$CREATING_CONVERSATION"),
        ).not.toBeInTheDocument(),
      { timeout: 3000 },
    );
    expect(screen.getAllByText(UNREACHABLE_MESSAGE)).toHaveLength(1);
    expect(input).toHaveTextContent(PROMPT);
  });

  it("names the missing MCP server in one readable toast and keeps the prompt when the launch is refused", async () => {
    // Arrange — Agent Server's structured 422 for a dangling MCP server ref.
    const detailMessage =
      "MCP server ref(s) not present in the user's MCP config: 'qa_mcp_a'";
    const body = {
      detail: {
        message: detailMessage,
        dangling_mcp_server_refs: ["qa_mcp_a"],
      },
    };
    vi.spyOn(AgentProfilesService, "getProfile").mockResolvedValue({
      profile: { name: "default" },
    } as never);
    vi.spyOn(
      AgentServerConversationService,
      "createConversation",
    ).mockRejectedValue(
      new HttpError(
        422,
        "Unprocessable Entity",
        body,
        `HTTP request failed (422 Unprocessable Entity): ${JSON.stringify(body)}`,
      ),
    );
    renderLauncher();

    // Act
    const input = await submitPrompt();

    // Assert
    await waitFor(() =>
      expect(screen.getAllByText(detailMessage)).toHaveLength(1),
    );
    expect(screen.queryByText(/dangling_mcp_server_refs/)).toBeNull();
    expect(input).toHaveTextContent(PROMPT);
  });
});
