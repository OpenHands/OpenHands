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

const mockNavigate = vi.hoisted(() => vi.fn());
const backendState = vi.hoisted(() => ({
  kind: "local" as "local" | "cloud",
  orgId: null as string | null,
}));

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({
    currentPath: "/",
    conversationId: null,
    isNavigating: false,
    navigate: mockNavigate,
  }),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: {
      id: "local-id",
      name: "Local",
      host: "http://localhost",
      apiKey: "test",
      kind: backendState.kind,
    },
    orgId: backendState.orgId,
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
    backendState.kind = "local";
    backendState.orgId = null;
    mockNavigate.mockClear();
    Object.defineProperty(HTMLElement.prototype, "innerText", {
      configurable: true,
      get() {
        return this.textContent ?? "";
      },
      set(value: string) {
        this.textContent = value;
      },
    });
    // <Toaster /> and framer-motion ask for the reduced-motion preference, which jsdom lacks.
    vi.stubGlobal(
      "matchMedia",
      vi.fn().mockReturnValue({
        matches: true,
        media: "(prefers-reduced-motion)",
        onchange: null,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        addListener: vi.fn(),
        removeListener: vi.fn(),
        dispatchEvent: vi.fn(),
      }),
    );
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

  it("shows one error toast, no loading toast and keeps the prompt when the agent server is unreachable", async () => {
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

    // Assert — a fast failure goes straight to a single error; the loading
    // toast never appears, even after its delay has passed.
    await waitFor(() =>
      expect(screen.getAllByText(UNREACHABLE_MESSAGE)).toHaveLength(1),
    );
    await new Promise((resolve) => {
      setTimeout(resolve, 800);
    });
    expect(
      screen.queryByText("HOME$CREATING_CONVERSATION"),
    ).not.toBeInTheDocument();
    expect(screen.getAllByText(UNREACHABLE_MESSAGE)).toHaveLength(1);
    expect(input).toHaveTextContent(PROMPT);
  });

  it("shows the loading toast only once a launch is slow, then replaces it with the error", async () => {
    // Arrange — the create stays in flight until the test fails it.
    vi.spyOn(AgentProfilesService, "getProfile").mockResolvedValue({
      profile: { name: "default" },
    } as never);
    let failCreate: (error: Error) => void = () => {};
    vi.spyOn(
      AgentServerConversationService,
      "createConversation",
    ).mockImplementation(
      () =>
        new Promise((_, reject) => {
          failCreate = reject;
        }),
    );
    renderLauncher();

    // Act
    const input = await submitPrompt();

    // Assert — nothing right away, the loading toast once the request lags,
    // then a single error in its place.
    expect(
      screen.queryByText("HOME$CREATING_CONVERSATION"),
    ).not.toBeInTheDocument();
    await screen.findByText(
      "HOME$CREATING_CONVERSATION",
      {},
      { timeout: 2000 },
    );
    failCreate(unreachable());
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

  it("keeps a slow launch's loading toast up briefly after navigating instead of flashing it", async () => {
    // Arrange — the create stays in flight until the test resolves it.
    vi.spyOn(AgentProfilesService, "getProfile").mockResolvedValue({
      profile: { name: "default" },
    } as never);
    let finishCreate: (value: unknown) => void = () => {};
    vi.spyOn(
      AgentServerConversationService,
      "createConversation",
    ).mockImplementation(
      () =>
        new Promise((resolve) => {
          finishCreate = resolve;
        }) as never,
    );
    // A dismissed toast stays in the DOM while it animates out, so watch the
    // dismissal itself.
    const dismissSpy = vi.spyOn(toast, "dismiss");
    renderLauncher();

    // Act — the create finishes right after the toast appears.
    await submitPrompt();
    await screen.findByText(
      "HOME$CREATING_CONVERSATION",
      {},
      { timeout: 2000 },
    );
    finishCreate({
      id: "conv-slow",
      status: "READY",
      app_conversation_id: "conv-slow",
      request: { initial_message: PROMPT, plugins: null },
    });

    // Assert — navigation is not held back, but the toast stays readable
    // for a moment before it goes.
    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith("/conversations/conv-slow"),
    );
    expect(dismissSpy).not.toHaveBeenCalled();
    await waitFor(() => expect(dismissSpy).toHaveBeenCalledTimes(1), {
      timeout: 2000,
    });
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
  it("shows a cloud prompt immediately and keeps a failed start inline until edited", async () => {
    backendState.kind = "cloud";
    let failCreate: (error: Error) => void = () => {};
    vi.spyOn(
      AgentServerConversationService,
      "createConversation",
    ).mockImplementation(
      () =>
        new Promise((_, reject) => {
          failCreate = reject;
        }),
    );
    const loadingToast = vi.spyOn(toast, "loading");
    renderLauncher();
    const input = await submitPrompt();
    expect(screen.getByTestId("startup-user-message")).toHaveTextContent(
      PROMPT,
    );
    expect(
      screen.getByTestId("conversation-startup-progress"),
    ).toHaveTextContent("HOME$CREATING_CONVERSATION");
    await waitFor(() =>
      expect(
        AgentServerConversationService.createConversation,
      ).toHaveBeenCalledOnce(),
    );
    failCreate(unreachable());
    await screen.findByRole("alert");
    expect(screen.getAllByText(UNREACHABLE_MESSAGE)).toHaveLength(1);
    expect(loadingToast).not.toHaveBeenCalled();
    expect(mockNavigate).toHaveBeenCalledTimes(1);
    expect(mockNavigate).toHaveBeenCalledWith(
      expect.stringMatching(/^\/conversations\/launch-/),
    );
    await userEvent
      .setup()
      .click(screen.getByRole("button", { name: "BUTTON$EDIT" }));
    await waitFor(() => expect(input).toHaveTextContent(PROMPT));
    expect(
      screen.queryByTestId("startup-user-message"),
    ).not.toBeInTheDocument();
  });

  it("does not redirect a late cloud launch after switching organizations", async () => {
    backendState.kind = "cloud";
    let finishCreate: (value: unknown) => void = () => {};
    vi.spyOn(
      AgentServerConversationService,
      "createConversation",
    ).mockImplementation(
      () =>
        new Promise((resolve) => {
          finishCreate = resolve;
        }) as never,
    );
    const view = renderLauncher();
    await submitPrompt();
    await waitFor(() =>
      expect(
        AgentServerConversationService.createConversation,
      ).toHaveBeenCalledOnce(),
    );
    backendState.orgId = "other-org";
    view.rerender(
      <QueryClientProvider client={createAgentServerQueryClient()}>
        <HomeChatLauncher />
        <Toaster />
      </QueryClientProvider>,
    );
    finishCreate({
      id: "task-late",
      status: "WORKING",
      app_conversation_id: null,
      request: {},
    });
    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(mockNavigate).toHaveBeenCalledTimes(1);
    expect(mockNavigate).toHaveBeenCalledWith(
      expect.stringMatching(/^\/conversations\/launch-/),
    );
    expect(
      screen.queryByTestId("startup-user-message"),
    ).not.toBeInTheDocument();
  });
});
