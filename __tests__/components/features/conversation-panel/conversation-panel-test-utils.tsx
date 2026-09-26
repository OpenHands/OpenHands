import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { I18nextProvider } from "react-i18next";
import i18n from "i18next";
import { NavigationProvider } from "#/context/navigation-context";
import { afterEach, beforeEach, vi } from "vitest";
import { createRoutesStub } from "react-router";
import React from "react";
import { renderWithProviders } from "test-utils";
import { ConversationPanel } from "#/components/features/conversation-panel/conversation-panel";
import { useConversationPanelPreferencesStore } from "#/stores/conversation-panel-preferences-store";
import { useArchivedConversationsStore } from "#/stores/archived-conversations-store";
import { usePinnedConversationsStore } from "#/stores/pinned-conversations-store";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { ExecutionStatus } from "#/types/agent-server/core";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { SEEDED_DEFAULT_BACKEND_ID } from "#/api/backend-registry/default-backend";
import type { Backend } from "#/api/backend-registry/types";

// Registered at module scope on purpose:  is hoisted per test file and
// must not sit inside .  Mirrors the placement main uses after
// 941acff, so splitting the suite does not regress that fix.
vi.mock("react-router", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-router")>()),
  Link: ({ children }: React.PropsWithChildren) => children,
  useNavigate: vi.fn(() => vi.fn()),
  useLocation: vi.fn(() => ({ pathname: "/conversation" })),
  useParams: vi.fn(() => ({ conversationId: "2" })),
}));

// Mock the unified stop conversation hook factory lives in each spec file
// (vi.mock must be declared alongside the spy it references). The shared
// beforeEach below clears all mocks, which covers that spy.

// Helper to create complete AppConversation mock data.
// Default timestamps use "now" so conversations are considered recent and
// rendered eagerly by the panel.  Each call produces a timestamp 1 s older
// than the previous one so that the sort-by-updated_at order matches the
// array insertion order (first created → newest → cards[0]).
let _mockConversationCounter = 0;
export const createMockConversation = (
  overrides: Partial<AppConversation> = {},
): AppConversation => {
  const ts = new Date(
    Date.now() - _mockConversationCounter++ * 1000,
  ).toISOString();
  return {
    id: "test-id",
    title: "Test Conversation",
    selected_repository: null,
    git_provider: null,
    selected_branch: null,
    updated_at: ts,
    created_at: ts,
    execution_status: ExecutionStatus.FINISHED,
    conversation_url: null,
    created_by_user_id: "user1",
    metrics: null,
    llm_model: null,
    trigger: null,
    pr_number: [],
    session_api_key: null,
    sandbox_id: null,
    sub_conversation_ids: [],
    ...overrides,
  };
};

export const onCloseMock = vi.fn();

export const RouterStub = createRoutesStub([
  {
    Component: () => <ConversationPanel onClose={onCloseMock} />,
    path: "/",
  },
  {
    // Add route to prevent "No routes matched location" warning
    Component: () => null,
    path: "/conversations/:conversationId",
  },
]);

export const renderConversationPanel = (
  options?: Parameters<typeof renderWithProviders>[1],
) => renderWithProviders(<RouterStub />, options);

// The old single-click filter menu is now the layouts menu, with the
// display toggles one level deeper in the Advanced options modal. The
// modal stays open across row clicks, so a second call is a no-op.
export const openAdvancedOptions = async (
  user: ReturnType<typeof userEvent.setup>,
) => {
  if (screen.queryByTestId("advanced-conversation-options-modal")) return;
  await user.click(screen.getByTestId("conversation-layouts-toggle"));
  await user.click(screen.getByTestId("advanced-options-row"));
};

export const cloudBackend: Backend = {
  id: "cloud-prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-key",
  kind: "cloud",
};

export const mockConversations: AppConversation[] = [
  createMockConversation({ id: "1", title: "Conversation 1" }),
  createMockConversation({ id: "2", title: "Conversation 2" }),
  createMockConversation({ id: "3", title: "Conversation 3" }),
];

// Wires the shared react-router mock plus the per-test store/API resets that
// every conversation-panel spec depends on. Call once inside each describe.
export const setupConversationPanelTest = () => {
  beforeEach(() => {
    vi.clearAllMocks();
    _mockConversationCounter = 0;
    usePinnedConversationsStore.setState({ pinsByBackendId: {} });
    useArchivedConversationsStore.setState({ archivesByBackendId: {} });
    useConversationPanelPreferencesStore.setState({
      showOlderConversations: true,
      olderConversationCutoff: "7d",
      showArchivedConversations: false,
      automationFilterMode: "all",
      selectedAutomationNames: [],
      selectedTagFacets: [],
      showTagsMetadata: false,
    });
    // Setup default mock for searchConversations
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [...mockConversations],
      next_page_id: null,
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    window.localStorage.clear();
    window.sessionStorage.clear();
    __resetActiveStoreForTests();
  });
};

// Re-exported render helper for specs that build a custom router stub.
export { render, QueryClient, QueryClientProvider, I18nextProvider, i18n, NavigationProvider, ActiveBackendProvider };
