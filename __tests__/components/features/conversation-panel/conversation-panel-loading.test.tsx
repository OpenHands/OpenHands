import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createRoutesStub } from "react-router";
import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { I18nextProvider } from "react-i18next";
import i18n from "i18next";
import { NavigationProvider } from "#/context/navigation-context";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { renderWithProviders } from "test-utils";
import { describe, expect, it, vi } from "vitest";
import {
  setupConversationPanelTest,
  createMockConversation,
  cloudBackend,
  mockConversations,
  onCloseMock,
  RouterStub,
  renderConversationPanel,
} from "./conversation-panel-test-utils";
import { ConversationPanel } from "#/components/features/conversation-panel/conversation-panel";
import { useConversationPanelPreferencesStore } from "#/stores/conversation-panel-preferences-store";
import { useArchivedConversationsStore } from "#/stores/archived-conversations-store";
import { usePinnedConversationsStore } from "#/stores/pinned-conversations-store";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { ExecutionStatus } from "#/types/agent-server/core";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { SEEDED_DEFAULT_BACKEND_ID } from "#/api/backend-registry/default-backend";

const mockStopConversationMutate = vi.fn();
vi.mock("#/hooks/mutation/use-unified-stop-conversation", () => ({
  useUnifiedPauseConversation: () => ({ mutate: mockStopConversationMutate }),
}));
vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
  TOAST_OPTIONS: {},
}));

describe("ConversationPanel list loading", () => {
  setupConversationPanelTest();

  it("pins the active backend onto each conversation link", async () => {
    // A tab opened with cmd/ctrl-click does not reliably inherit the opener's
    // sessionStorage, so the link has to carry the backend it belongs to or
    // the new tab resolves the conversation against whichever backend
    // localStorage happens to hold.
    renderConversationPanel();
    const cards = await screen.findAllByTestId("conversation-card");

    const href = cards[0].closest("a")?.getAttribute("href");
    expect(href).toBe(`/conversations/1?backend=${SEEDED_DEFAULT_BACKEND_ID}`);
  });

  it("should render the conversations", async () => {
    renderConversationPanel();
    const cards = await screen.findAllByTestId("conversation-card");

    // NOTE that we filter out conversations that don't have a created_at property
    // (mock data has 4 conversations, but only 3 have a created_at property)
    expect(cards).toHaveLength(3);
  });

  it("includes the active backend scope in conversation card links", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-2" });

    const ScopedRouterStub = createRoutesStub([
      {
        Component: () => <ConversationPanel onClose={onCloseMock} />,
        path: "/",
      },
      {
        Component: () => null,
        path: "/conversations/:conversationId",
      },
    ]);
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    render(
      <QueryClientProvider client={queryClient}>
        <I18nextProvider i18n={i18n}>
          <ActiveBackendProvider>
            <NavigationProvider
              value={{
                currentPath: "/",
                conversationId: null,
                isNavigating: false,
                navigate: vi.fn(),
              }}
            >
              <ScopedRouterStub />
            </NavigationProvider>
          </ActiveBackendProvider>
        </I18nextProvider>
      </QueryClientProvider>,
    );

    const title = await screen.findByText("Conversation 1");
    expect(title.closest("a")).toHaveAttribute(
      "href",
      "/conversations/1?backend=cloud-prod&org=org-2",
    );
  });

  it("includes the active backend scope in compact conversation row links", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-2" });
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        createMockConversation({
          id: "running",
          title: "Running Conversation",
          execution_status: ExecutionStatus.RUNNING,
        }),
      ],
      next_page_id: null,
    });

    const CompactRouterStub = createRoutesStub([
      {
        Component: () => <ConversationPanel compact />,
        path: "/",
      },
      {
        Component: () => null,
        path: "/conversations/:conversationId",
      },
    ]);
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    render(
      <QueryClientProvider client={queryClient}>
        <I18nextProvider i18n={i18n}>
          <ActiveBackendProvider>
            <NavigationProvider
              value={{
                currentPath: "/",
                conversationId: null,
                isNavigating: false,
                navigate: vi.fn(),
              }}
            >
              <CompactRouterStub />
            </NavigationProvider>
          </ActiveBackendProvider>
        </I18nextProvider>
      </QueryClientProvider>,
    );

    expect(
      await screen.findByLabelText("Running Conversation"),
    ).toHaveAttribute(
      "href",
      "/conversations/running?backend=cloud-prod&org=org-2",
    );
  });

  it("should display an empty state when there are no conversations", async () => {
    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy.mockResolvedValue({
      items: [],
      next_page_id: null,
    });

    renderConversationPanel();

    const emptyState = await screen.findByText("CONVERSATION$NO_CONVERSATIONS");
    expect(emptyState).toBeInTheDocument();
  });

  it("keeps load more available when the visible list is empty and another page exists", async () => {
    // Client-side filters (archiving, thread scope) can hide every row of the
    // loaded pages. Hiding "Load more" there would strand the remaining
    // backend pages behind an empty list with no way to reach them.
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [],
      next_page_id: "page-2",
    });

    renderConversationPanel();

    await screen.findByText("CONVERSATION$NO_CONVERSATIONS");
    expect(screen.getByTestId("load-more-conversations")).toBeInTheDocument();
  });

  it("can reach an unarchived conversation on the next page after archiving every loaded row", async () => {
    // Archiving filters rows out of the visible list without changing
    // backend pagination. If every currently loaded conversation is archived
    // while hasNextPage is still true, Load more must stay available and
    // fetching the next page must surface the unarchived conversation.
    const user = userEvent.setup();
    const page1 = [
      createMockConversation({ id: "archived-1", title: "Archived 1" }),
      createMockConversation({ id: "archived-2", title: "Archived 2" }),
    ];
    const page2 = [
      createMockConversation({
        id: "visible-next",
        title: "Unarchived on next page",
      }),
    ];
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockImplementation(async (_limit, pageId) => {
      if (pageId === "page-2") {
        return { items: page2, next_page_id: null };
      }
      return { items: page1, next_page_id: "page-2" };
    });

    useArchivedConversationsStore.setState({
      archivesByBackendId: {
        "default-local": ["archived-1", "archived-2"],
      },
    });
    useConversationPanelPreferencesStore.setState({
      showArchivedConversations: false,
    });

    renderConversationPanel();

    await screen.findByText("CONVERSATION$NO_CONVERSATIONS");
    expect(
      screen.queryByText("Unarchived on next page"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByTestId("load-more-conversations"));

    expect(
      await screen.findByText("Unarchived on next page"),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("CONVERSATION$NO_CONVERSATIONS"),
    ).not.toBeInTheDocument();
  });

  it("does not flash the loading skeleton during a background refetch when the list is empty", async () => {
    // Arrange: first call (initial load) resolves with an empty list.
    // Second call (the background refetch) is kept in-flight so we can
    // observe the UI while `isFetching` is true — the exact window in
    // which the buggy `isFetching`-gated code flashed the skeleton.
    let resolveRefetch:
      | ((value: { items: never[]; next_page_id: null }) => void)
      | undefined;
    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy
      .mockResolvedValueOnce({ items: [], next_page_id: null })
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveRefetch = resolve;
          }),
      );

    // Use a QueryClient we can reach so we can trigger the background
    // refetch directly. This drives the same code path as the hook's 10s
    // `refetchInterval` (both flip `isFetching` to true while existing
    // data stays in the cache) without paying the cost of fake-timer
    // gymnastics around React Query's async state machine.
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const PanelRouterStub = createRoutesStub([
      {
        Component: () => <ConversationPanel />,
        path: "/",
      },
      {
        Component: () => null,
        path: "/conversations/:conversationId",
      },
    ]);
    render(
      <QueryClientProvider client={queryClient}>
        <I18nextProvider i18n={i18n}>
          <NavigationProvider
            value={{
              currentPath: "/",
              conversationId: "test-conversation-id",
              isNavigating: false,
              navigate: vi.fn(),
            }}
          >
            <PanelRouterStub />
          </NavigationProvider>
        </I18nextProvider>
      </QueryClientProvider>,
    );

    // Wait for the initial fetch to settle into the empty state.
    expect(
      await screen.findByText("CONVERSATION$NO_CONVERSATIONS"),
    ).toBeInTheDocument();

    // Act: trigger a background refetch directly on the cached query.
    // This drives the same code path as the hook's 10s `refetchInterval`
    // (both flip `isFetching` to true while the cached data stays
    // intact). The second mock holds the request in-flight so the
    // in-flight UI is observable. We fire-and-forget the fetch because
    // awaiting it would hang on the held promise.
    const conversationsQuery = queryClient
      .getQueryCache()
      .getAll()
      .find(
        (query) =>
          query.queryKey[0] === "user" && query.queryKey[1] === "conversations",
      );
    if (!conversationsQuery) {
      throw new Error("conversations query was not registered");
    }
    await act(async () => {
      void conversationsQuery.fetch();
      // Yield once so React Query can dispatch the in-flight state.
      await Promise.resolve();
    });

    // Assert: the background refetch fired, but the skeleton did not
    // flicker back in.
    await waitFor(() => {
      expect(searchConversationsSpy).toHaveBeenCalledTimes(2);
    });
    expect(
      screen.queryByTestId("conversation-card-skeleton"),
    ).not.toBeInTheDocument();

    // Settle the in-flight refetch so React Query can clean up.
    resolveRefetch?.({ items: [], next_page_id: null });
  });

  it("should not display the empty state when there are no conversations and the panel is compact", async () => {
    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy.mockResolvedValue({
      items: [],
      next_page_id: null,
    });

    const CompactRouterStub = createRoutesStub([
      {
        Component: () => <ConversationPanel compact />,
        path: "/",
      },
      {
        Component: () => null,
        path: "/conversations/:conversationId",
      },
    ]);

    renderWithProviders(<CompactRouterStub />);

    await waitFor(() => {
      expect(
        screen.queryByTestId("conversation-card-skeleton-compact"),
      ).not.toBeInTheDocument();
    });

    expect(
      screen.queryByText("CONVERSATION$NO_CONVERSATIONS"),
    ).not.toBeInTheDocument();
  });

  it("hides closed conversations in compact mode", async () => {
    vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    ).mockResolvedValue({
      items: [
        createMockConversation({
          id: "running",
          title: "Running Conversation",
          execution_status: ExecutionStatus.RUNNING,
        }),
        createMockConversation({
          id: "closed",
          title: "Closed Conversation",
          execution_status: ExecutionStatus.PAUSED,
        }),
      ],
      next_page_id: null,
    });

    const CompactRouterStub = createRoutesStub([
      {
        Component: () => <ConversationPanel compact />,
        path: "/",
      },
      {
        Component: () => null,
        path: "/conversations/:conversationId",
      },
    ]);

    renderWithProviders(<CompactRouterStub />);

    expect(
      await screen.findByLabelText("Running Conversation"),
    ).toBeInTheDocument();
    expect(
      screen.queryByLabelText("Closed Conversation"),
    ).not.toBeInTheDocument();
  });

  it("should not render fetch errors in the conversation panel", async () => {
    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy.mockRejectedValue(
      new Error("Failed to fetch conversations"),
    );

    renderConversationPanel();

    await waitFor(() => {
      expect(
        screen.queryByText("Failed to fetch conversations"),
      ).not.toBeInTheDocument();
    });
  });

  it("should call onClose after clicking a card", async () => {
    const user = userEvent.setup();
    renderConversationPanel();
    const cards = await screen.findAllByTestId("conversation-card");
    const firstCard = cards[1];

    await user.click(firstCard);

    expect(onCloseMock).toHaveBeenCalledOnce();
  });

  it("should refetch data on rerenders", async () => {
    const user = userEvent.setup();
    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy.mockResolvedValue({
      items: [...mockConversations],
      next_page_id: null,
    });

    function PanelWithToggle() {
      const [isOpen, setIsOpen] = React.useState(true);
      return (
        <>
          <button type="button" onClick={() => setIsOpen((prev) => !prev)}>
            Toggle
          </button>
          {isOpen && <ConversationPanel onClose={onCloseMock} />}
        </>
      );
    }

    const MyRouterStub = createRoutesStub([
      {
        Component: PanelWithToggle,
        path: "/",
      },
    ]);

    renderWithProviders(<MyRouterStub />);

    const toggleButton = screen.getByText("Toggle");

    // Initial render
    const cards = await screen.findAllByTestId("conversation-card");
    expect(cards).toHaveLength(3);

    // Toggle off
    await user.click(toggleButton);
    expect(screen.queryByTestId("conversation-card")).not.toBeInTheDocument();

    // Toggle on
    await user.click(toggleButton);
    const newCards = await screen.findAllByTestId("conversation-card");
    expect(newCards).toHaveLength(3);
  });

  it("keeps invalid timestamps recent and shows older conversations by default", async () => {
    const now = Date.now();
    const minutesAgo = (minutes: number) =>
      new Date(now - minutes * 60 * 1000).toISOString();

    const user = userEvent.setup();
    const searchConversationsSpy = vi.spyOn(
      AgentServerConversationService,
      "searchConversations",
    );
    searchConversationsSpy.mockReset();
    searchConversationsSpy
      .mockResolvedValueOnce({
        items: [
          createMockConversation({
            id: "recent",
            title: "Recent Conversation",
            updated_at: minutesAgo(59),
          }),
          createMockConversation({
            id: "invalid",
            title: "Invalid Timestamp",
            updated_at: "invalid-date",
          }),
          createMockConversation({
            id: "missing",
            title: "Missing Timestamp",
            updated_at: undefined as unknown as string,
          }),
          createMockConversation({
            id: "older",
            title: "Older Conversation",
            updated_at: minutesAgo(61),
          }),
        ],
        next_page_id: "page-2",
      })
      .mockResolvedValueOnce({
        items: [
          createMockConversation({
            id: "paged",
            title: "Paged Conversation",
            updated_at: minutesAgo(30),
          }),
        ],
        next_page_id: null,
      });

    renderConversationPanel();

    expect(await screen.findByText("Recent Conversation")).toBeInTheDocument();
    expect(screen.getByText("Invalid Timestamp")).toBeInTheDocument();
    expect(screen.getByText("Missing Timestamp")).toBeInTheDocument();
    expect(screen.getByText("Older Conversation")).toBeInTheDocument();
    expect(
      screen.getByTestId("older-conversations-summary"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("load-more-conversations")).toBeInTheDocument();

    await user.click(screen.getByTestId("load-more-conversations"));

    await waitFor(() => {
      expect(searchConversationsSpy).toHaveBeenCalledWith(20, "page-2");
    });
    expect(await screen.findByText("Paged Conversation")).toBeInTheDocument();
  });

  describe("active conversation highlight", () => {
    it("marks the currently active conversation with data-active=true", async () => {
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [
          createMockConversation({ id: "1", title: "Conversation 1" }),
          createMockConversation({ id: "2", title: "Conversation 2" }),
          createMockConversation({ id: "3", title: "Conversation 3" }),
        ],
        next_page_id: null,
      });

      renderWithProviders(<RouterStub />, {
        navigation: { conversationId: "2", currentPath: "/conversations/2" },
      });

      const cards = await screen.findAllByTestId("conversation-card");
      expect(cards).toHaveLength(3);
      expect(cards[0]).toHaveAttribute("data-active", "false");
      expect(cards[1]).toHaveAttribute("data-active", "true");
      expect(cards[2]).toHaveAttribute("data-active", "false");
    });

    it("renders no active card when no conversation is selected", async () => {
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: [createMockConversation({ id: "1", title: "Conversation 1" })],
        next_page_id: null,
      });

      renderWithProviders(<RouterStub />, {
        navigation: { conversationId: null, currentPath: "/" },
      });

      const cards = await screen.findAllByTestId("conversation-card");
      expect(cards[0]).toHaveAttribute("data-active", "false");
    });
  });

  describe("pinned conversations", () => {
    it("shows a pinned section above the conversations list when pins exist", async () => {
      usePinnedConversationsStore
        .getState()
        .pinConversation("default-local", "2");

      renderConversationPanel();

      const pinnedSection = await screen.findByTestId(
        "conversation-panel-pinned-section",
      );
      expect(
        within(pinnedSection).getByText("CONVERSATION_PANEL$PINNED"),
      ).toBeInTheDocument();
      expect(
        within(pinnedSection).getAllByTestId("conversation-card"),
      ).toHaveLength(1);
      expect(
        within(pinnedSection).getByTestId("conversation-pin-toggle-2"),
      ).toBeInTheDocument();
    });

    it("renders pinned conversations only in the pinned section in chronological mode", async () => {
      usePinnedConversationsStore
        .getState()
        .pinConversation("default-local", "2");

      renderConversationPanel();

      const pinnedSection = await screen.findByTestId(
        "conversation-panel-pinned-section",
      );
      expect(
        within(pinnedSection).getAllByTestId("conversation-card"),
      ).toHaveLength(1);
      expect(await screen.findAllByTestId("conversation-card")).toHaveLength(3);
      expect(screen.getAllByText("Conversation 2")).toHaveLength(1);
    });

    it("renders pinned conversations only in the pinned section in grouped mode", async () => {
      useConversationPanelPreferencesStore.setState({ organizeMode: "grouped" });
      usePinnedConversationsStore
        .getState()
        .pinConversation("default-local", "2");

      renderConversationPanel();

      const pinnedSection = await screen.findByTestId(
        "conversation-panel-pinned-section",
      );
      expect(
        within(pinnedSection).getAllByTestId("conversation-card"),
      ).toHaveLength(1);
      expect(await screen.findAllByTestId("conversation-card")).toHaveLength(3);
      expect(screen.getAllByText("Conversation 2")).toHaveLength(1);
    });

    it("hides the pinned section after the last pin is removed", async () => {
      usePinnedConversationsStore
        .getState()
        .pinConversation("default-local", "2");

      const user = userEvent.setup();
      renderConversationPanel();

      const pinnedSection = await screen.findByTestId(
        "conversation-panel-pinned-section",
      );
      const pinnedCard = within(pinnedSection).getByTestId("conversation-card");
      await user.hover(pinnedCard);
      await user.click(
        within(pinnedSection).getByTestId("conversation-pin-toggle-2"),
      );

      await waitFor(() => {
        expect(
          screen.queryByTestId("conversation-panel-pinned-section"),
        ).not.toBeInTheDocument();
      });
    });

    it("shows only five pinned conversations before a More control", async () => {
      const manyConversations = Array.from({ length: 6 }, (_, index) =>
        createMockConversation({
          id: String(index + 1),
          title: `Conversation ${index + 1}`,
        }),
      );
      vi.spyOn(
        AgentServerConversationService,
        "searchConversations",
      ).mockResolvedValue({
        items: manyConversations,
        next_page_id: null,
      });
      for (const conversation of manyConversations) {
        usePinnedConversationsStore
          .getState()
          .pinConversation("default-local", conversation.id);
      }

      renderConversationPanel();

      const pinnedSection = await screen.findByTestId(
        "conversation-panel-pinned-section",
      );
      expect(
        within(pinnedSection).getAllByTestId("conversation-card"),
      ).toHaveLength(5);
      expect(
        within(pinnedSection).getByTestId("conversation-panel-pinned-view-more"),
      ).toHaveTextContent("CONVERSATION_PANEL$MORE");
    });
  });
});
