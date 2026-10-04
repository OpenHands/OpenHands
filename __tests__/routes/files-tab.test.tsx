import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { MemoryRouter } from "react-router";

import FilesTab from "#/routes/files-tab";
import { useFilesTabStore } from "#/stores/files-tab-store";
import { NavigationProvider } from "#/context/navigation-context";
import {
  LOCAL_STORAGE_KEYS,
  setConversationState,
} from "#/utils/conversation-local-storage";

// Mocks must be declared before the SUT is imported.
const useWorkspaceFilesMock = vi.fn();
const useWorkspaceFileContentMock = vi.fn();
const useActiveConversationMock = vi.fn();

vi.mock("#/hooks/query/use-workspace-files", () => ({
  useWorkspaceFiles: () => useWorkspaceFilesMock(),
}));

vi.mock("#/hooks/query/use-workspace-file-content", () => ({
  useWorkspaceFileContent: (path: string | null) =>
    useWorkspaceFileContentMock(path),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));

function renderTab(conversationId: string | null = null) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <MemoryRouter>
      <NavigationProvider
        value={{
          currentPath: "/",
          conversationId,
          isNavigating: false,
          navigate: () => {},
        }}
      >
        <QueryClientProvider client={client}>
          <FilesTab />
        </QueryClientProvider>
      </NavigationProvider>
    </MemoryRouter>,
  );
}

function openFile(path: string, conversationId: string | null = null) {
  useFilesTabStore.getState().setSelectedPath(path, conversationId);
}

describe("FilesTab", () => {
  beforeEach(() => {
    useFilesTabStore.setState({
      selectedPath: null,
      selectedConversationId: null,
      openPaths: [],
    });
    localStorage.clear();

    useWorkspaceFilesMock.mockReset();
    useWorkspaceFileContentMock.mockReset();
    useActiveConversationMock.mockReset();

    useWorkspaceFilesMock.mockReturnValue({
      data: ["index.html", "src/main.ts", "README.md"],
      isLoading: false,
    });
    useWorkspaceFileContentMock.mockReturnValue({
      data: {
        path: "index.html",
        kind: "text",
        text: "<!doctype html><html><body>hello</body></html>",
        staticUrl:
          "http://localhost:3000/api/conversations/c1/workspace/index.html",
        mimeType: "text/html",
      },
      isLoading: false,
      isError: false,
    });
    useActiveConversationMock.mockReturnValue({
      data: {
        workspace: { working_dir: "/workspace/project" },
      },
    });
  });

  it("renders the file browser without a Diff/Commits toggle", () => {
    renderTab();

    expect(screen.getByTestId("files-tab")).toBeInTheDocument();
    expect(
      screen.queryByTestId("files-tab-content-mode-toggle"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("files-tab-diff-toggle"),
    ).not.toBeInTheDocument();
  });

  it("does not open file tabs until a file is selected", () => {
    renderTab();

    expect(useWorkspaceFileContentMock).toHaveBeenCalledWith(null);
    expect(screen.queryByRole("tab")).not.toBeInTheDocument();
    expect(
      screen.getByTestId("file-quick-row-tree-toggle"),
    ).toBeInTheDocument();
  });

  it("shows the active conversation workspace path", () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        workspace: { working_dir: "/workspace/project/worktree-123" },
      },
    });

    renderTab();

    expect(
      screen.getByTestId("files-tab-workspace-path-value"),
    ).toHaveTextContent("/workspace/project/worktree-123");
  });

  it("opens a tab when a file is selected and closes it from the tab strip", async () => {
    const user = userEvent.setup();
    openFile("src/main.ts");
    renderTab();

    expect(
      screen.getByTestId("file-quick-row-item-src/main.ts"),
    ).toBeInTheDocument();
    expect(screen.getByRole("tab", { selected: true })).toHaveTextContent(
      "main.ts",
    );

    await user.click(screen.getByTestId("file-quick-row-close-src/main.ts"));

    expect(
      screen.queryByTestId("file-quick-row-item-src/main.ts"),
    ).not.toBeInTheDocument();
    expect(useFilesTabStore.getState().selectedPath).toBeNull();
    expect(useFilesTabStore.getState().openPaths).toEqual([]);
  });

  it("keeps vertical edges on every open tab", () => {
    openFile("README.md");
    openFile("src/main.ts");
    renderTab();

    const firstTab = screen.getByTestId(
      "file-quick-row-item-README.md",
    ).parentElement;
    const secondTab = screen.getByTestId(
      "file-quick-row-item-src/main.ts",
    ).parentElement;
    expect(firstTab).toHaveClass("border-l");
    expect(firstTab).toHaveClass("border-r");
    expect(secondTab).toHaveClass("border-r");
  });

  it("renders the binary fallback in plain mode for binary files", async () => {
    useWorkspaceFileContentMock.mockReturnValue({
      data: {
        path: "logo.png",
        kind: "binary",
        text: null,
        staticUrl:
          "http://localhost:3000/api/conversations/c1/workspace/logo.png",
        mimeType: "application/octet-stream",
      },
      isLoading: false,
      isError: false,
    });
    const user = userEvent.setup();

    openFile("logo.png");
    renderTab();

    await user.click(
      screen.getByTestId("files-tab-content-mode-toggle-option-plain"),
    );

    expect(
      screen.getByTestId("file-content-viewer-binary-fallback"),
    ).toBeInTheDocument();
  });

  it("shows the file name (not the full path) on quick-row tabs", () => {
    openFile("src/main.ts");
    renderTab();

    const tab = screen.getByTestId("file-quick-row-item-src/main.ts");
    expect(tab).toHaveTextContent("main.ts");
    expect(tab).toHaveAttribute("title", "src/main.ts");
    expect(tab).toHaveAttribute("role", "tab");
  });

  it("shows the file tree by default and collapses it via the caret", async () => {
    const user = userEvent.setup();

    renderTab();

    expect(screen.getByTestId("files-tab-tree")).toBeInTheDocument();

    await user.click(screen.getByTestId("file-quick-row-tree-toggle"));
    expect(screen.queryByTestId("files-tab-tree")).not.toBeInTheDocument();

    await user.click(screen.getByTestId("file-quick-row-tree-toggle"));
    expect(screen.getByTestId("files-tab-tree")).toBeInTheDocument();
  });

  it("exposes a grippable resize handle on the tree's right edge when expanded", () => {
    window.localStorage.clear();

    renderTab();

    expect(
      screen.getByTestId("files-tab-tree-resize-handle"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("files-tab-tree")).toHaveStyle({
      width: "224px",
    });
  });

  it("opens a tab from the file tree when a file is clicked", async () => {
    const user = userEvent.setup();
    renderTab();

    await user.click(screen.getByTestId("file-tree-file-README.md"));

    expect(useFilesTabStore.getState().openPaths).toContain("README.md");
    expect(
      screen.getByTestId("file-quick-row-item-README.md"),
    ).toBeInTheDocument();
  });

  it("renders markdown content via MarkdownRenderer in rich mode", async () => {
    useWorkspaceFilesMock.mockReturnValue({
      data: ["README.md"],
      isLoading: false,
    });
    useWorkspaceFileContentMock.mockReturnValue({
      data: {
        path: "README.md",
        kind: "text",
        text: "# Hello\n\nSome **bold** text",
        staticUrl:
          "http://localhost:3000/api/conversations/c1/workspace/README.md",
        mimeType: "text/markdown",
      },
      isLoading: false,
      isError: false,
    });

    openFile("README.md");
    renderTab();

    await waitFor(() => {
      expect(
        screen.getByTestId("file-content-viewer-markdown"),
      ).toBeInTheDocument();
    });

    expect(
      screen.getByRole("heading", { level: 1, name: "Hello" }),
    ).toBeInTheDocument();
    expect(screen.getByText("bold").tagName.toLowerCase()).toBe("strong");
  });

  it("shows highlighted source (not rich markdown) when toggled to plain on a .md", async () => {
    useWorkspaceFilesMock.mockReturnValue({
      data: ["README.md"],
      isLoading: false,
    });
    useWorkspaceFileContentMock.mockReturnValue({
      data: {
        path: "README.md",
        kind: "text",
        text: "# Hello\n\nSome **bold** text",
        staticUrl:
          "http://localhost:3000/api/conversations/c1/workspace/README.md",
        mimeType: "text/markdown",
      },
      isLoading: false,
      isError: false,
    });
    const user = userEvent.setup();

    openFile("README.md");
    renderTab();

    await user.click(
      screen.getByTestId("files-tab-content-mode-toggle-option-plain"),
    );

    const highlighted = await screen.findByTestId(
      "file-content-viewer-highlighted",
    );
    expect(highlighted.getAttribute("data-language")).toBe("markdown");
    expect(
      screen.queryByRole("heading", { level: 1, name: "Hello" }),
    ).not.toBeInTheDocument();
  });

  it("uses the workspace fileserver URL as the iframe src for HTML files", async () => {
    useWorkspaceFilesMock.mockReturnValue({
      data: ["index.html"],
      isLoading: false,
    });
    const staticUrl =
      "https://agent.example.com/api/conversations/conv-1/workspace/index.html";
    useWorkspaceFileContentMock.mockReturnValue({
      data: {
        path: "index.html",
        kind: "text",
        text: "<!doctype html><body>hi</body>",
        staticUrl,
        mimeType: "text/html",
      },
      isLoading: false,
      isError: false,
    });

    openFile("index.html");
    renderTab();

    const iframe = await screen.findByTestId("file-content-viewer-iframe");
    expect(iframe).toHaveAttribute("src", `${staticUrl}?v=0`);
    expect(iframe).toHaveAttribute("sandbox", "allow-same-origin");
  });

  it("switches between rich and plain content modes", async () => {
    useWorkspaceFilesMock.mockReturnValue({
      data: ["src/main.ts"],
      isLoading: false,
    });
    useWorkspaceFileContentMock.mockReturnValue({
      data: {
        path: "src/main.ts",
        kind: "text",
        text: "console.log('hi');",
        staticUrl:
          "http://localhost:3000/api/conversations/c1/workspace/src/main.ts",
        mimeType: "text/plain",
      },
      isLoading: false,
      isError: false,
    });
    const user = userEvent.setup();

    openFile("src/main.ts");
    renderTab();

    await user.click(
      screen.getByTestId("files-tab-content-mode-toggle-option-plain"),
    );
    const highlighted = await screen.findByTestId(
      "file-content-viewer-highlighted",
    );
    expect(highlighted).toBeInTheDocument();
    expect(highlighted.getAttribute("data-language")).toBe("typescript");
  });

  it("shows refresh on the file quick-row", () => {
    openFile("index.html");
    renderTab();

    const quickRow = screen.getByTestId("file-quick-row");
    const refresh = screen.getByTestId("files-tab-refresh");
    expect(quickRow).toContainElement(refresh);
    expect(refresh).toHaveAttribute("aria-label", "FILES$REFRESH");
  });

  it("persists and restores open files plus tree visibility across remount", async () => {
    const conversationId = "persist-files-conv";
    const user = userEvent.setup();
    const { unmount } = renderTab(conversationId);

    await user.click(screen.getByTestId("file-quick-row-tree-toggle"));
    openFile("README.md", conversationId);
    openFile("src/main.ts", conversationId);

    await waitFor(() => {
      const raw = localStorage.getItem(
        `${LOCAL_STORAGE_KEYS.CONVERSATION_STATE}-${conversationId}`,
      );
      expect(raw).toBeTruthy();
      const stored = JSON.parse(raw as string);
      expect(stored.filesTabTreeVisible).toBe(false);
      expect(stored.filesTabOpenPaths).toEqual(["README.md", "src/main.ts"]);
      expect(stored.filesTabSelectedPath).toBe("src/main.ts");
    });

    // Simulate a full reload: empty in-memory store, remount the tab.
    unmount();
    useFilesTabStore.setState({
      selectedPath: null,
      selectedConversationId: null,
      openPaths: [],
    });
    renderTab(conversationId);

    await waitFor(() => {
      expect(useFilesTabStore.getState().openPaths).toEqual([
        "README.md",
        "src/main.ts",
      ]);
    });
    expect(screen.getByRole("tab", { selected: true })).toHaveTextContent(
      "main.ts",
    );
    expect(screen.queryByTestId("files-tab-tree")).not.toBeInTheDocument();
  });

  it("hydrates open files from conversation localStorage on first mount", async () => {
    const conversationId = "hydrate-files-conv";
    setConversationState(conversationId, {
      filesTabOpenPaths: ["README.md"],
      filesTabSelectedPath: "README.md",
      filesTabTreeVisible: true,
    });

    renderTab(conversationId);

    await waitFor(() => {
      expect(
        screen.getByTestId("file-quick-row-item-README.md"),
      ).toBeInTheDocument();
    });
    expect(screen.getByTestId("files-tab-tree")).toBeInTheDocument();
  });

  it("shows the Rich/Plain toggle only when a file is open", async () => {
    renderTab();

    expect(
      screen.queryByTestId("files-tab-content-mode-toggle"),
    ).not.toBeInTheDocument();

    openFile("index.html");

    const modeToggle = await screen.findByTestId(
      "files-tab-content-mode-toggle",
    );
    const content = screen.getByTestId("files-tab-content");
    expect(content).toContainElement(modeToggle);
    expect(content).toContainElement(
      screen.getByTestId("files-tab-open-in-new-window"),
    );
  });

  describe("conversation switching", () => {
    it("ignores a stale selection from another conversation", async () => {
      useFilesTabStore.setState({
        selectedPath: "demo.html",
        selectedConversationId: "conv-a",
        openPaths: ["demo.html"],
      });
      useWorkspaceFilesMock.mockReturnValue({
        data: ["app.html"],
        isLoading: false,
      });

      renderTab("conv-b");

      await waitFor(() => {
        expect(useWorkspaceFileContentMock).toHaveBeenCalledWith(null);
      });
      expect(useWorkspaceFileContentMock).not.toHaveBeenCalledWith("demo.html");
      expect(screen.queryByRole("tab")).not.toBeInTheDocument();
    });

    it("preserves a selection that belongs to the current conversation", async () => {
      useFilesTabStore.setState({
        selectedPath: "report.html",
        selectedConversationId: "conv-b",
        openPaths: ["report.html"],
      });
      useWorkspaceFilesMock.mockReturnValue({
        data: ["report.html", "app.html"],
        isLoading: false,
      });

      renderTab("conv-b");

      await waitFor(() => {
        expect(useWorkspaceFileContentMock).toHaveBeenCalledWith("report.html");
      });
      expect(useFilesTabStore.getState().selectedPath).toBe("report.html");
      expect(
        screen.getByTestId("file-quick-row-item-report.html"),
      ).toBeInTheDocument();
    });

    it("clears open tabs when the tab switches to a different conversation", async () => {
      openFile("demo.html", "conv-a");
      useWorkspaceFilesMock.mockReturnValue({
        data: ["demo.html"],
        isLoading: false,
      });

      const { rerender } = renderTab("conv-a");

      await waitFor(() => {
        expect(useFilesTabStore.getState().selectedPath).toBe("demo.html");
      });
      expect(useFilesTabStore.getState().openPaths).toEqual(["demo.html"]);

      useWorkspaceFileContentMock.mockClear();
      useWorkspaceFilesMock.mockReturnValue({
        data: ["app.html"],
        isLoading: false,
      });
      const client = new QueryClient({
        defaultOptions: { queries: { retry: false } },
      });
      rerender(
        <MemoryRouter>
          <NavigationProvider
            value={{
              currentPath: "/",
              conversationId: "conv-b",
              isNavigating: false,
              navigate: () => {},
            }}
          >
            <QueryClientProvider client={client}>
              <FilesTab />
            </QueryClientProvider>
          </NavigationProvider>
        </MemoryRouter>,
      );

      await waitFor(() => {
        expect(useFilesTabStore.getState().selectedPath).toBeNull();
      });
      expect(useFilesTabStore.getState().openPaths).toEqual([]);
      expect(useFilesTabStore.getState().selectedConversationId).toBe("conv-b");
      expect(useWorkspaceFileContentMock).not.toHaveBeenCalledWith("demo.html");
    });
  });
});

describe("useFilesTabStore open tabs", () => {
  beforeEach(() => {
    useFilesTabStore.setState({
      selectedPath: null,
      selectedConversationId: null,
      openPaths: [],
    });
  });

  it("appends newly opened paths and selects a neighbor on close", () => {
    const { setSelectedPath, closeOpenPath } = useFilesTabStore.getState();

    setSelectedPath("a.ts", "c1");
    setSelectedPath("b.ts", "c1");
    setSelectedPath("c.ts", "c1");
    expect(useFilesTabStore.getState().openPaths).toEqual([
      "a.ts",
      "b.ts",
      "c.ts",
    ]);

    closeOpenPath("b.ts");
    expect(useFilesTabStore.getState().openPaths).toEqual(["a.ts", "c.ts"]);
    // Closing the active middle tab selects the right neighbor.
    expect(useFilesTabStore.getState().selectedPath).toBe("c.ts");

    closeOpenPath("c.ts");
    expect(useFilesTabStore.getState().selectedPath).toBe("a.ts");
  });
});
