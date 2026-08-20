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
  useOptionalConversationId: () => ({ conversationId: "test-conversation-id" }),
  useConversationId: () => ({
    conversationId: mockConversationId,
  }),
}));



// Mock lazy-loaded components
vi.mock("#/routes/files-tab", () => ({
  default: () => <div data-testid="files-tab-content">Files Tab Content</div>,
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
    useConversationStore.setState({ selectedTab: "files" });
    // Reset conversation ID
    mockConversationId = "test-conversation-id-123";
  });

  afterEach(() => {
    vi.clearAllMocks();
    queryClient.clear();
  });

  describe("Rendering", () => {
    it("should render files tab content by default", async () => {
      setSelectedTab("files");

      render(<ConversationTabContent />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByTestId("files-tab-content")).toBeInTheDocument();
      });
    });

    it("should render files tab when selectedTab is null", async () => {
      setSelectedTab(null);

      render(<ConversationTabContent />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByTestId("files-tab-content")).toBeInTheDocument();
      });
    });
  });

  describe("Tab switching", () => {
    it("should render browser tab when selected", async () => {
      setSelectedTab("browser");

      render(<ConversationTabContent />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByTestId("browser-tab-content")).toBeInTheDocument();
      });
    });

    it("should render terminal tab when selected", async () => {
      setSelectedTab("terminal");

      render(<ConversationTabContent />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByTestId("terminal-tab-content")).toBeInTheDocument();
      });
    });

    it("should render planner tab when selected", async () => {
      setSelectedTab("planner");

      render(<ConversationTabContent />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByTestId("planner-tab-content")).toBeInTheDocument();
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
      // First render with files tab
      setSelectedTab("files");

      const { rerender } = render(<ConversationTabContent />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(screen.getByTestId("files-tab-content")).toBeInTheDocument();
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
      setSelectedTab("files");

      render(<ConversationTabContent />, { wrapper: createWrapper() });

      // The component should render without throwing
      await waitFor(() => {
        expect(screen.getByTestId("files-tab-content")).toBeInTheDocument();
      });
    });
  });
});
