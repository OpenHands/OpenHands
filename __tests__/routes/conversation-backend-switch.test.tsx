import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import type { SharedConversation } from "@openhands/typescript-client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { NavigationProvider } from "#/context/navigation-context";
import ConversationView from "#/routes/conversation";
import type { Backend } from "#/api/backend-registry/types";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { getCloudSharedConversation } from "#/api/cloud/shared-conversation-service.api";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

// Mock the underlying service the conversation queries depend on.
vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: { batchGetAppConversations: vi.fn() },
  }),
);

// Stub the heavy presentational subtree so the test focuses on the route's
// teardown behaviour, not the conversation UI.
vi.mock(
  "#/components/features/conversation/conversation-main/conversation-main",
  () => ({
    ConversationMain: () => <div data-testid="conversation-main" />,
  }),
);
vi.mock(
  "#/components/features/conversation/conversation-main/conversation-mobile-panel-page",
  () => ({
    ConversationMobilePanelPage: () => <div data-testid="conversation-panel" />,
  }),
);
vi.mock("#/wrapper/event-handler", () => ({
  EventHandler: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  ),
}));
vi.mock("#/contexts/websocket-provider-wrapper", () => ({
  WebSocketProviderWrapper: ({ children }: { children: React.ReactNode }) => (
    <>{children}</>
  ),
}));
vi.mock("#/hooks/query/use-task-polling", () => ({
  useTaskPollingController: () => ({
    isTask: false,
    taskStatus: null,
    taskDetail: null,
  }),
}));
vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => ({ data: true }),
}));
vi.mock("#/api/cloud/conversation-service.api", () => ({
  resumeCloudSandbox: vi.fn(),
}));
// Mock the cloud sharing service the read-only fallback probes.
vi.mock("#/api/cloud/shared-conversation-service.api", () => ({
  getCloudSharedConversation: vi.fn(),
  searchCloudSharedEvents: vi.fn(),
}));
vi.mock("#/utils/custom-toast-handlers", async (importOriginal) => ({
  ...(await importOriginal<typeof import("#/utils/custom-toast-handlers")>()),
  displayErrorToast: vi.fn(),
}));

const localBackend: Backend = {
  id: "local-1",
  name: "Local 1",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-key",
  kind: "cloud",
};

const CLOUD_CONVERSATION_ID = "conv-cloud";

function makeConversation(id: string): AppConversation {
  return {
    id,
    created_by_user_id: null,
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    title: "Test",
    trigger: null,
    pr_number: [],
    llm_model: null,
    metrics: null,
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    execution_status: null,
    sandbox_status: null,
    conversation_url: "https://sandbox.example.com/api",
    session_api_key: null,
    sandbox_id: null,
    sub_conversation_ids: [],
  };
}

function renderConversation() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <MemoryRouter initialEntries={[`/conversations/${CLOUD_CONVERSATION_ID}`]}>
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>
          <NavigationProvider
            value={{
              currentPath: `/conversations/${CLOUD_CONVERSATION_ID}`,
              conversationId: CLOUD_CONVERSATION_ID,
              isNavigating: false,
              navigate: vi.fn(),
            }}
          >
            <ConversationView />
          </NavigationProvider>
        </ActiveBackendProvider>
      </QueryClientProvider>
    </MemoryRouter>,
  );
}

function makeSharedConversation(id: string): SharedConversation {
  return {
    id,
    created_by_user_id: "creator-1",
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    title: "Automated code review",
    pr_number: [],
    llm_model: null,
    metrics: null,
    parent_conversation_id: null,
    sub_conversation_ids: [],
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
  };
}

// Render the route inside a router so the fallback's navigation lands on
// observable destinations.
function renderConversationRoute(conversationId: string) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <MemoryRouter initialEntries={[`/conversations/${conversationId}`]}>
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>
          <NavigationProvider
            value={{
              currentPath: `/conversations/${conversationId}`,
              conversationId,
              isNavigating: false,
              navigate: vi.fn(),
            }}
          >
            <Routes>
              <Route
                path="/conversations/:conversationId"
                element={<ConversationView />}
              />
              <Route
                path="/shared/conversations/:conversationId"
                element={<div data-testid="shared-conversation-view" />}
              />
              <Route
                path="/conversations"
                element={<div data-testid="conversations-home" />}
              />
            </Routes>
          </NavigationProvider>
        </ActiveBackendProvider>
      </QueryClientProvider>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  vi.mocked(getCloudSharedConversation).mockReset();
  vi.mocked(displayErrorToast).mockReset();
  vi.mocked(
    AgentServerConversationService.batchGetAppConversations,
  ).mockReset();
  vi.mocked(
    AgentServerConversationService.batchGetAppConversations,
  ).mockResolvedValue([makeConversation(CLOUD_CONVERSATION_ID)]);
  setRegisteredBackends([localBackend, cloudBackend]);
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("conversation route — backend switch", () => {
  it("tears down the conversation view when the active backend changes mid-conversation", async () => {
    // Arrange — the cloud conversation renders while the cloud backend is active.
    setActiveSelection({ backendId: cloudBackend.id });
    renderConversation();
    expect(await screen.findByTestId("conversation-main")).toBeInTheDocument();

    // Act — switch to the local backend without leaving the conversation.
    setActiveSelection({ backendId: localBackend.id });

    // Assert — the subtree unmounts instead of rendering the cloud
    // conversation under the local backend, so its per-conversation queries
    // cannot fire against a backend the id is foreign to.
    await waitFor(() => {
      expect(screen.queryByTestId("conversation-main")).not.toBeInTheDocument();
    });
  });
});

describe("conversation route — shared read-only fallback", () => {
  it("sends an org member to the read-only shared view when only the shared lookup can see the conversation", async () => {
    // Arrange — the owner lookup misses on cloud, but the conversation is
    // shared with this user (an automation conversation from their org).
    setActiveSelection({ backendId: cloudBackend.id });
    vi.mocked(
      AgentServerConversationService.batchGetAppConversations,
    ).mockResolvedValue([null]);
    vi.mocked(getCloudSharedConversation).mockResolvedValue(
      makeSharedConversation(CLOUD_CONVERSATION_ID),
    );

    // Act
    renderConversationRoute(CLOUD_CONVERSATION_ID);

    // Assert
    expect(
      await screen.findByTestId("shared-conversation-view"),
    ).toBeInTheDocument();
    expect(getCloudSharedConversation).toHaveBeenCalledWith(
      CLOUD_CONVERSATION_ID,
    );
    expect(displayErrorToast).not.toHaveBeenCalled();
  });

  it("reports the conversation as missing when neither lookup can see it on cloud", async () => {
    // Arrange
    setActiveSelection({ backendId: cloudBackend.id });
    vi.mocked(
      AgentServerConversationService.batchGetAppConversations,
    ).mockResolvedValue([null]);
    vi.mocked(getCloudSharedConversation).mockResolvedValue(null);

    // Act
    renderConversationRoute(CLOUD_CONVERSATION_ID);

    // Assert
    expect(await screen.findByTestId("conversations-home")).toBeInTheDocument();
    expect(displayErrorToast).toHaveBeenCalledTimes(1);
  });

  it("does not consult the shared lookup on a local backend", async () => {
    // Arrange
    setActiveSelection({ backendId: localBackend.id });
    vi.mocked(
      AgentServerConversationService.batchGetAppConversations,
    ).mockResolvedValue([null]);

    // Act
    renderConversationRoute(CLOUD_CONVERSATION_ID);

    // Assert
    expect(await screen.findByTestId("conversations-home")).toBeInTheDocument();
    expect(getCloudSharedConversation).not.toHaveBeenCalled();
    expect(displayErrorToast).toHaveBeenCalledTimes(1);
  });
});
