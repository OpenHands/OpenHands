import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router";
import { ConversationTabContent } from "#/components/features/conversation/conversation-tabs/conversation-tab-content/conversation-tab-content";
import {
  useConversationStore,
  ConversationTab,
} from "#/stores/conversation-store";

// Mock useConversationId hook
let mockConversationId = "test-conversation-id-123";
vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({
    conversationId: mockConversationId,
  }),
}));

// Mock useUnifiedGetGitChanges hook (used by ConversationTabTitle)
vi.mock("#/hooks/query/use-unified-get-git-changes", () => ({
  useUnifiedGetGitChanges: () => ({
    refetch: vi.fn(),
    data: [],
    isLoading: false,
  }),
}));

// Mock lazy-loaded components
vi.mock("#/routes/changes-tab", () => ({
  default: () => <div data-testid="editor-tab-content">Editor Tab Content</div>,
}));

// Control for lazy loading test
let pendingBrowserTab: { promise: Promise<void>; resolve: () => void } | null = null;
vi.mock("#/routes/browser-tab", () => ({
  default: () => {
    if (pendingBrowserTab) {
      throw pendingBrowserTab.promise;
    }
    return <div data-testid="browser-tab-content">Browser Tab Content</div>;
  },
}));

vi.mock("#/routes/served-tab", () => ({
  default: () => (
    <div data-testid="served-tab-content">Served Tab Content</div>
  ),
}));

vi.mock("#/routes/vscode-tab", () => ({
  default: () => (
    <div data-testid="vscode-tab-content">VSCode Tab Content</div>
  ),
}));

vi.mock("#/routes/planner-tab", () => ({
  default: () => (
    <div data-testid="planner-tab-content">Planner Tab Content</div>
  ),
}));

vi.mock("#/components/features/terminal/terminal", () => ({
  default: () => (
    <div data-testid="terminal-tab-content">Terminal Tab Content</div>
  ),
}));

// Mock ConversationLoading component
vi.mock("#/components/features/conversation/conversation-loading", () => ({
  ConversationLoading: () => (
    <div data-testid="conversation-loading">Loading...</div>
  ),
}));

describe("ConversationTabContent", () => {
  let queryClient: QueryClient;

  const createWrapper = () => {
    return ({ children }: { children: React.ReactNode }) => (
      <MemoryRouter initialEntries={["/conversations/test-conversation-id"]}>
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
      </MemoryRouter>
    );
  };

  const setSelectedTab = (tab: ConversationTab | null) => {
    useConversationStore.setState({ selectedTab: tab });
  };

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });
    // Reset store state
    useConversationStore.setState({ selectedTab: "editor" });
    // Reset conversation ID
    mockConversationId = "test-conversation-id-123";
  });

  afterEach(() => {
    vi.clearAllMocks();
    queryClient.clear();
  });

  describe("Rendering", () => {
    it("should render the container with correct structure", async () => {
      render(<ConversationTabContent />, { wrapper: createWrapper() });

      // Should show the title for the default tab (editor -> COMMON$CHANGES)
      await waitFor(() => {
        expect(screen.getByText("COMMON$CHANGES")).toBeInTheDocument();
      });
    });

    it("should render editor tab content by default", async () => {
      setSelectedTab("editor");

      render(<ConversationTabContent />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByTestId("editor-tab-content")).toBeInTheDocument();
      });

      expect(screen.getByText("COMMON$CHANGES")).toBeInTheDocument();
    });

    it("should render editor tab when selectedTab is null", async () => {
      setSelectedTab(null);

      render(<ConversationTabContent />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByTestId("editor-tab-content")).toBeInTheDocument();
      });

      expect(screen.getByText("COMMON$CHANGES")).toBeInTheDocument();
    });
  });

  describe("Tab switching", () => {
    it("should render browser tab when selected", async () => {
      setSelectedTab("browser");

      render(<ConversationTabContent />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByTestId("browser-tab-content")).toBeInTheDocument();
      });

      expect(screen.getByText("COMMON$BROWSER")).toBeInTheDocument();
    });

    it("should render served tab when selected", async () => {
      setSelectedTab("served");

      render(<ConversationTabContent />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByTestId("served-tab-content")).toBeInTheDocument();
      });

      expect(screen.getByText("COMMON$APP")).toBeInTheDocument();
    });

    it("should render vscode tab when selected", async () => {
      setSelectedTab("vscode");

      render(<ConversationTabContent />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByTestId("vscode-tab-content")).toBeInTheDocument();
      });

      expect(screen.getByText("COMMON$CODE")).toBeInTheDocument();
    });

    it("should render terminal tab when selected", async () => {
      setSelectedTab("terminal");

      render(<ConversationTabContent />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByTestId("terminal-tab-content")).toBeInTheDocument();
      });

      expect(screen.getByText("COMMON$TERMINAL")).toBeInTheDocument();
    });

    it("should render planner tab when selected", async () => {
      setSelectedTab("planner");

      render(<ConversationTabContent />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByTestId("planner-tab-content")).toBeInTheDocument();
      });

      expect(screen.getByText("COMMON$PLANNER")).toBeInTheDocument();
    });
  });

  describe("Title display", () => {
    const tabTitleMapping: Array<{
      tab: ConversationTab;
      expectedTitle: string;
    }> = [
        { tab: "editor", expectedTitle: "COMMON$CHANGES" },
        { tab: "browser", expectedTitle: "COMMON$BROWSER" },
        { tab: "served", expectedTitle: "COMMON$APP" },
        { tab: "vscode", expectedTitle: "COMMON$CODE" },
        { tab: "terminal", expectedTitle: "COMMON$TERMINAL" },
        { tab: "planner", expectedTitle: "COMMON$PLANNER" },
      ];

    tabTitleMapping.forEach(({ tab, expectedTitle }) => {
      it(`should display "${expectedTitle}" title for "${tab}" tab`, async () => {
        setSelectedTab(tab);

        render(<ConversationTabContent />, { wrapper: createWrapper() });

        await waitFor(() => {
          expect(screen.getByText(expectedTitle)).toBeInTheDocument();
        });
      });
    });
  });

  describe("Tab key behavior", () => {
    it("should remount terminal when conversation ID changes", async () => {
      setSelectedTab("terminal");

      const { rerender } = render(<ConversationTabContent />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(screen.getByTestId("terminal-tab-content")).toBeInTheDocument();
      });

      // Get reference to the terminal DOM node
      const terminalBefore = screen.getByTestId("terminal-tab-content");

      // Change conversation ID
      mockConversationId = "test-conversation-id-456";

      // Rerender to pick up the new conversation ID
      rerender(<ConversationTabContent />);

      await waitFor(() => {
        expect(screen.getByTestId("terminal-tab-content")).toBeInTheDocument();
      });

      // Get new reference
      const terminalAfter = screen.getByTestId("terminal-tab-content");

      // If key includes conversation ID, component should remount = different DOM node
      expect(terminalBefore).not.toBe(terminalAfter);
    });

    it("should NOT remount non-terminal tabs when conversation ID changes", async () => {
      setSelectedTab("browser");

      const { rerender } = render(<ConversationTabContent />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(screen.getByTestId("browser-tab-content")).toBeInTheDocument();
      });

      // Get reference to the browser DOM node
      const browserBefore = screen.getByTestId("browser-tab-content");

      // Change conversation ID
      mockConversationId = "test-conversation-id-789";

      // Rerender
      rerender(<ConversationTabContent />);

      await waitFor(() => {
        expect(screen.getByTestId("browser-tab-content")).toBeInTheDocument();
      });

      // Get new reference
      const browserAfter = screen.getByTestId("browser-tab-content");

      // If key does NOT include conversation ID, component should NOT remount = same DOM node
      expect(browserBefore).toBe(browserAfter);
    });
  });

  describe("Lazy loading", () => {
    afterEach(() => {
      pendingBrowserTab = null;
    });

    it("should show loading fallback while component is loading", async () => {
      let resolveFn: () => void;
      pendingBrowserTab = {
        promise: new Promise<void>((resolve) => {
          resolveFn = resolve;
        }),
        resolve: () => {
          pendingBrowserTab = null; // Clear first so re-render doesn't throw again
          resolveFn();
        },
      };

      setSelectedTab("browser");

      render(<ConversationTabContent />, { wrapper: createWrapper() });

      // Verify loading fallback is shown
      expect(screen.getByTestId("conversation-loading")).toBeInTheDocument();

      // Resolve to load the component
      pendingBrowserTab!.resolve();

      // Verify content appears
      await waitFor(() => {
        expect(screen.getByTestId("browser-tab-content")).toBeInTheDocument();
      });
    });
  });

  describe("Tab state persistence", () => {
    it("should render content based on store state", async () => {
      // First render with editor tab
      setSelectedTab("editor");

      const { rerender } = render(<ConversationTabContent />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(screen.getByTestId("editor-tab-content")).toBeInTheDocument();
      });

      // Change the store state
      setSelectedTab("terminal");

      // Rerender
      rerender(<ConversationTabContent />);

      await waitFor(() => {
        expect(screen.getByTestId("terminal-tab-content")).toBeInTheDocument();
      });
    });
  });

  describe("Suspense boundary", () => {
    it("should wrap tab content in Suspense boundary", async () => {
      setSelectedTab("editor");

      render(<ConversationTabContent />, { wrapper: createWrapper() });

      // The component should render without throwing
      await waitFor(() => {
        expect(screen.getByTestId("editor-tab-content")).toBeInTheDocument();
      });
    });
  });
});
