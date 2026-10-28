import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { compile } from "@tailwindcss/node";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { MemoryRouter } from "react-router";

import FilesTab from "#/routes/files-tab";
import { useFilesTabStore } from "#/stores/files-tab-store";
import { useWorkspaceMutationCounter } from "#/stores/use-workspace-mutation-counter";
import { NavigationProvider } from "#/context/navigation-context";
import { FILES_TAB_TREE_WIDTH_STORAGE_KEY } from "#/components/features/files-tab/files-tab-tree.constants";
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

const { downloadFileMock, displayErrorToastMock } = vi.hoisted(() => ({
  downloadFileMock: vi.fn(),
  displayErrorToastMock: vi.fn(),
}));

vi.mock("#/api/runtime-service/agent-server-runtime-service", () => ({
  default: { downloadFile: downloadFileMock },
}));

vi.mock("#/utils/custom-toast-handlers", async (importOriginal) => ({
  ...(await importOriginal<typeof import("#/utils/custom-toast-handlers")>()),
  displayErrorToast: displayErrorToastMock,
}));

const createObjectURL = vi.fn();
const revokeObjectURL = vi.fn();

// jsdom does not implement object URLs.
function stubObjectUrls() {
  let created = 0;
  createObjectURL.mockReset().mockImplementation(() => {
    created += 1;
    return `blob:http://localhost/file-${created}`;
  });
  revokeObjectURL.mockReset();
  vi.stubGlobal(
    "URL",
    class extends URL {
      static createObjectURL = createObjectURL;

      static revokeObjectURL = revokeObjectURL;
    },
  );
}

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

// Follow the rendered classes into production Tailwind CSS instead of assuming
// the utility name or custom property that connects the reader to the writer.
async function getTreeWidthProperty(tree: HTMLElement) {
  const compiler = await compile(
    await readFile(resolve("src/tailwind.css"), "utf8"),
    { base: resolve("src"), onDependency: () => {} },
  );
  const css = compiler.build([...tree.classList]);
  const declarations = [...css.matchAll(/\.((?:\\.|[\w-])+)\s*\{([^}]+)\}/g)]
    .filter(([, selector]) =>
      tree.classList.contains(selector.replace(/\\(.)/g, "$1")),
    )
    .map(([, , body]) => body)
    .join("\n");
  const width = /(?:^|\n)\s*width:\s*var\((--[\w-]+)\);/.exec(declarations);
  expect(width).not.toBeNull();
  return width![1];
}

describe("FilesTab", () => {
  beforeEach(() => {
    useFilesTabStore.setState({
      selectedPath: null,
      selectedConversationId: null,
      openPaths: [],
    });
    useWorkspaceMutationCounter.setState({ count: 0 });
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

  // @spec WFD-001 — Configurable local workspace discovery
  it("shows an incomplete-tree message when discovery is truncated", () => {
    useWorkspaceFilesMock.mockReturnValue({
      data: ["a.txt", "b.txt"],
      isLoading: false,
      isTruncated: true,
    });
    renderTab();
    expect(screen.getByRole("status")).toHaveTextContent(
      "FILES$DISCOVERY_TRUNCATED",
    );
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

  it("retains resized pixel widths through clamps, collapse and remount", async () => {
    // Arrange: the real resize hook measures the layout's left edge.
    const user = userEvent.setup();
    openFile("README.md");
    const view = renderTab();
    const tree = screen.getByTestId("files-tab-tree");
    const widthProperty = await getTreeWidthProperty(tree);
    vi.spyOn(tree.parentElement!, "getBoundingClientRect").mockReturnValue(
      DOMRect.fromRect({ x: 100, y: 0, width: 1000, height: 600 }),
    );
    const handle = screen.getByTestId("files-tab-tree-resize-handle");

    // Act + Assert: pointer coordinates update pixel width and both clamps.
    fireEvent.mouseDown(handle.firstElementChild!);
    expect(document.querySelector("[data-panel-drag-shield]")).not.toBeNull();
    for (const [clientX, expectedWidth] of [
      [424, 324],
      [1000, 480],
      [100, 160],
      [424, 324],
    ]) {
      fireEvent.mouseMove(document, { clientX });
      expect(tree.style.getPropertyValue(widthProperty)).toBe(
        `${expectedWidth}px`,
      );
    }
    fireEvent.mouseUp(document);
    expect(document.querySelector("[data-panel-drag-shield]")).toBeNull();
    expect(
      JSON.parse(localStorage.getItem(FILES_TAB_TREE_WIDTH_STORAGE_KEY)!),
    ).toBe(324);

    // Collapse/restore and a new mount retain width and the selected file.
    await user.click(screen.getByTestId("file-quick-row-tree-toggle"));
    expect(screen.queryByTestId("files-tab-tree")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("files-tab-tree-resize-handle"),
    ).not.toBeInTheDocument();
    await user.click(screen.getByTestId("file-quick-row-tree-toggle"));
    expect(
      screen
        .getByTestId("files-tab-tree")
        .style.getPropertyValue(widthProperty),
    ).toBe("324px");
    view.unmount();
    renderTab();
    expect(
      screen
        .getByTestId("files-tab-tree")
        .style.getPropertyValue(widthProperty),
    ).toBe("324px");
    expect(screen.getByRole("tab", { selected: true })).toHaveTextContent(
      "README.md",
    );
  });

  it("applies the default tree width through a compiled CSS reader", async () => {
    // Arrange + Act
    renderTab();
    const tree = screen.getByTestId("files-tab-tree");
    const widthProperty = await getTreeWidthProperty(tree);

    // Assert: native CSS consumes a px value without host spacing/font tokens.
    expect(tree.style.width).toBe("");
    expect(tree.style.getPropertyValue(widthProperty)).toBe("224px");
    expect(
      screen.getByTestId("files-tab-tree-resize-handle"),
    ).toBeInTheDocument();
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

  it("re-requests the rich preview and the new-window link on Refresh", async () => {
    useWorkspaceFilesMock.mockReturnValue({
      data: ["index.html"],
      isLoading: false,
    });
    const staticUrl = `${window.location.origin}/api/conversations/conv-1/workspace/index.html`;
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
    const user = userEvent.setup();

    openFile("index.html");
    renderTab();
    const iframe = await screen.findByTestId("file-content-viewer-iframe");
    const srcBefore = iframe.getAttribute("src");

    await user.click(screen.getByTestId("files-tab-refresh"));

    // A new URL is the only way to make an <iframe>/<img> skip a cached
    // response, so Refresh must change it (#17921).
    await waitFor(() => {
      expect(
        screen.getByTestId("file-content-viewer-iframe").getAttribute("src"),
      ).not.toBe(srcBefore);
    });
    expect(
      screen.getByTestId("files-tab-open-in-new-window").getAttribute("href"),
    ).toBe(
      screen.getByTestId("file-content-viewer-iframe").getAttribute("src"),
    );
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

  // Cloud backends hand back the file as a `data:` URL, which browsers refuse
  // to open as a top-level page (OHE-3735).
  describe("open in new window on a Cloud backend", () => {
    function cloudMarkdownFile(text: string) {
      return {
        data: {
          path: "README.md",
          kind: "text",
          text,
          staticUrl: `data:text/markdown;charset=utf-8;base64,${btoa(text)}`,
          mimeType: "text/markdown",
        },
        isLoading: false,
        isError: false,
      };
    }

    beforeEach(stubObjectUrls);

    it("opens the file's text as plain text", async () => {
      // Arrange
      useWorkspaceFileContentMock.mockReturnValue(
        cloudMarkdownFile("# Hello\n\nSome **bold** text"),
      );

      // Act
      openFile("README.md");
      renderTab();

      // Assert
      const link = await screen.findByTestId("files-tab-open-in-new-window");
      expect(link).toHaveAttribute("href", "blob:http://localhost/file-1");
      const blob: Blob = createObjectURL.mock.calls[0][0];
      expect(blob.type).toBe("text/plain;charset=utf-8");
      await expect(blob.text()).resolves.toBe("# Hello\n\nSome **bold** text");
    });

    it("opens the current file body after Refresh", async () => {
      // Arrange
      useWorkspaceFileContentMock.mockReturnValue(cloudMarkdownFile("# Old"));
      const user = userEvent.setup();
      openFile("README.md");
      renderTab();
      await screen.findByTestId("files-tab-open-in-new-window");
      useWorkspaceFileContentMock.mockReturnValue(cloudMarkdownFile("# New"));

      // Act
      await user.click(screen.getByTestId("files-tab-refresh"));

      // Assert
      await waitFor(() => {
        expect(
          screen.getByTestId("files-tab-open-in-new-window"),
        ).toHaveAttribute("href", "blob:http://localhost/file-2");
      });
      await expect(createObjectURL.mock.calls[1][0].text()).resolves.toBe(
        "# New",
      );
      expect(revokeObjectURL).toHaveBeenCalledWith(
        "blob:http://localhost/file-1",
      );
    });

    it("offers no link for a file without decoded text", () => {
      // Arrange
      useWorkspaceFileContentMock.mockReturnValue({
        data: {
          path: "logo.png",
          kind: "image",
          text: null,
          staticUrl: "data:image/png;base64,iVBORw0KGgo=",
          mimeType: "image/png",
        },
        isLoading: false,
        isError: false,
      });

      // Act
      openFile("logo.png");
      renderTab();

      // Assert
      expect(
        screen.getByTestId("files-tab-content-mode-toggle"),
      ).toBeInTheDocument();
      expect(
        screen.queryByTestId("files-tab-open-in-new-window"),
      ).not.toBeInTheDocument();
    });
  });

  describe("open in new window on a backend at another origin", () => {
    const remoteBase =
      "https://backend.example.com/api/conversations/conv-1/workspace";

    function remoteFile(
      path: string,
      kind: string,
      mimeType: string,
      text: string | null = null,
    ) {
      return {
        data: {
          path,
          kind,
          text,
          staticUrl: `${remoteBase}/${path}`,
          mimeType,
        },
        isLoading: false,
        isError: false,
      };
    }

    beforeEach(() => {
      stubObjectUrls();
      downloadFileMock.mockReset();
      displayErrorToastMock.mockReset();
      useActiveConversationMock.mockReturnValue({
        data: {
          id: "conv-1",
          conversation_url:
            "https://backend.example.com/api/conversations/conv-1",
          session_api_key: "session-key",
          execution_status: "idle",
          workspace: { working_dir: "/workspace/project" },
        },
      });
    });

    it("opens a text file's decoded source as plain text", async () => {
      // Arrange
      useWorkspaceFileContentMock.mockReturnValue(
        remoteFile(
          "index.html",
          "text",
          "text/html",
          "<script>alert(1)</script>",
        ),
      );

      // Act
      openFile("index.html");
      renderTab();

      // Assert
      const link = await screen.findByTestId("files-tab-open-in-new-window");
      expect(link).toHaveAttribute("href", "blob:http://localhost/file-1");
      const blob: Blob = createObjectURL.mock.calls[0][0];
      expect(blob.type).toBe("text/plain;charset=utf-8");
      await expect(blob.text()).resolves.toBe("<script>alert(1)</script>");
      expect(downloadFileMock).not.toHaveBeenCalled();
    });

    it.each([
      ["logo.png", "image", "image/png", "image/png"],
      ["report.pdf", "pdf", "application/pdf", "application/pdf"],
      [
        "data.bin",
        "binary",
        "application/octet-stream",
        "application/octet-stream",
      ],
      ["icon.svg", "image", "image/svg+xml", "text/plain;charset=utf-8"],
    ])(
      "opens %s (%s) from the header-authenticated download",
      async (path, kind, mimeType, blobType) => {
        // Arrange
        const bytes = new Uint8Array([1, 2, 3]).buffer;
        downloadFileMock.mockResolvedValue(bytes);
        useWorkspaceFileContentMock.mockReturnValue(
          remoteFile(path, kind, mimeType),
        );

        // Act
        openFile(path);
        renderTab();

        // Assert
        const link = await screen.findByTestId("files-tab-open-in-new-window");
        expect(link).toHaveAttribute("href", "blob:http://localhost/file-1");
        expect(downloadFileMock).toHaveBeenCalledWith(
          "https://backend.example.com/api/conversations/conv-1",
          "session-key",
          `/workspace/project/${path}`,
        );
        const blob: Blob = createObjectURL.mock.calls[0][0];
        expect(blob.type).toBe(blobType);
        expect(new Uint8Array(await blob.arrayBuffer())).toEqual(
          new Uint8Array(bytes),
        );
      },
    );

    it("shows an error and offers no backend link when the download fails", async () => {
      // Arrange
      downloadFileMock.mockRejectedValue(new Error("401"));
      useWorkspaceFileContentMock.mockReturnValue(
        remoteFile("logo.png", "image", "image/png"),
      );

      // Act
      openFile("logo.png");
      renderTab();

      // Assert
      await waitFor(() => {
        expect(displayErrorToastMock).toHaveBeenCalledWith("FILES$LOAD_ERROR");
      });
      expect(
        screen.queryByTestId("files-tab-open-in-new-window"),
      ).not.toBeInTheDocument();
    });

    it("shows the download error once while the same file fails again", async () => {
      // Arrange
      downloadFileMock.mockRejectedValue(new Error("404"));
      useWorkspaceFileContentMock.mockReturnValue(
        remoteFile("logo.png", "image", "image/png"),
      );
      const user = userEvent.setup();
      openFile("logo.png");
      renderTab();
      await waitFor(() => {
        expect(displayErrorToastMock).toHaveBeenCalled();
      });

      // Act
      await user.click(screen.getByTestId("files-tab-refresh"));

      // Assert
      await waitFor(() => {
        expect(downloadFileMock).toHaveBeenCalledTimes(2);
      });
      expect(displayErrorToastMock).toHaveBeenCalledTimes(1);
    });

    it("shows the download error again after the user comes back to the file", async () => {
      // Arrange
      downloadFileMock.mockRejectedValue(new Error("404"));
      useWorkspaceFileContentMock.mockImplementation((path: string | null) =>
        path === "README.md"
          ? remoteFile("README.md", "text", "text/markdown", "# Hi")
          : remoteFile("logo.png", "image", "image/png"),
      );
      const user = userEvent.setup();
      openFile("README.md");
      openFile("logo.png");
      renderTab();
      await waitFor(() => {
        expect(displayErrorToastMock).toHaveBeenCalledTimes(1);
      });

      // Act
      await user.click(screen.getByTestId("file-quick-row-item-README.md"));
      await user.click(screen.getByTestId("file-quick-row-item-logo.png"));

      // Assert
      await waitFor(() => {
        expect(displayErrorToastMock).toHaveBeenCalledTimes(2);
      });
    });

    it("does not download while the runtime is not ready", () => {
      // Arrange
      useActiveConversationMock.mockReturnValue({
        data: {
          id: "conv-1",
          conversation_url:
            "https://backend.example.com/api/conversations/conv-1",
          session_api_key: "session-key",
          execution_status: "paused",
          workspace: { working_dir: "/workspace/project" },
        },
      });
      useWorkspaceFileContentMock.mockReturnValue(
        remoteFile("logo.png", "image", "image/png"),
      );

      // Act
      openFile("logo.png");
      renderTab();

      // Assert
      expect(downloadFileMock).not.toHaveBeenCalled();
    });

    it("revokes the object URL when the Files tab unmounts", async () => {
      // Arrange
      downloadFileMock.mockResolvedValue(new Uint8Array([1]).buffer);
      useWorkspaceFileContentMock.mockReturnValue(
        remoteFile("logo.png", "image", "image/png"),
      );
      openFile("logo.png");
      const { unmount } = renderTab();
      await screen.findByTestId("files-tab-open-in-new-window");

      // Act
      unmount();

      // Assert
      expect(revokeObjectURL).toHaveBeenCalledWith(
        "blob:http://localhost/file-1",
      );
    });
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
