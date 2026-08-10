import React from "react";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FileContentViewer } from "#/components/features/files-tab/file-content-viewer";
import type { ViewMode } from "#/components/features/files-tab/view-mode";
import { useWorkspaceMutationCounter } from "#/stores/use-workspace-mutation-counter";

// Mock the *services* the file-content hook depends on — not the hook itself —
// so the real classification (text decoded, then flipped to binary on a NUL
// sniff) runs end to end through the viewer.
const useWorkspaceSessionMock = vi.fn();
vi.mock("#/hooks/query/use-workspace-session", async (importOriginal) => {
  const real =
    await importOriginal<
      typeof import("#/hooks/query/use-workspace-session")
    >();
  return {
    ...real, // keep the real joinWorkspaceUrl the hook builds its fetch URL with
    useWorkspaceSession: () => useWorkspaceSessionMock(),
  };
});

const useActiveConversationMock = vi.fn();
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));

const useRuntimeIsReadyMock = vi.fn();
vi.mock("#/hooks/use-runtime-is-ready", () => ({
  useRuntimeIsReady: () => useRuntimeIsReadyMock(),
}));

const getActiveBackendMock = vi.fn();
vi.mock("#/api/backend-registry/active-store", () => ({
  getActiveBackend: () => getActiveBackendMock(),
}));

// The hook statically imports the cloud runtime service; stub the module so
// this local-path test never loads the real cloud/proxy machinery. The test
// uses the fetch (local) path, so downloadFile is never called or asserted.
vi.mock("#/api/runtime-service/agent-server-runtime-service", () => ({
  default: { downloadFile: vi.fn() },
}));

const fetchMock = vi.fn();

const BASE_URL =
  "https://agent.example.com/api/conversations/conv-1/workspace/";

function renderViewer(path: string, viewMode: ViewMode = "rich") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <FileContentViewer path={path} viewMode={viewMode} />
    </QueryClientProvider>,
  );
}

describe("FileContentViewer", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", fetchMock);
    fetchMock.mockReset();
    useWorkspaceSessionMock.mockReset();
    useActiveConversationMock.mockReset();
    useRuntimeIsReadyMock.mockReset();
    getActiveBackendMock.mockReset();

    useRuntimeIsReadyMock.mockReturnValue(true);
    useActiveConversationMock.mockReturnValue({
      data: {
        id: "conv-1",
        conversation_url: "https://agent.example.com/api/conversations/conv-1",
        session_api_key: "session-key",
      },
    });
    useWorkspaceSessionMock.mockReturnValue({
      data: { baseUrl: BASE_URL },
      isLoading: false,
      isError: false,
      error: null,
    });
    getActiveBackendMock.mockReturnValue({
      backend: { id: "local-1", kind: "local", host: "http://localhost:8000" },
      orgId: null,
    });
    useWorkspaceMutationCounter.setState({ count: 0 });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  // The acceptance criteria require the clear message in BOTH view modes. The
  // plain-mode fallback and the rich-mode binary branch both route through
  // UnpreviewableFallback, so one parametrized spec covers both code paths.
  it.each(["rich", "plain"] as const)(
    "shows a clear unsupported-document message for an Office file (.pptx) in %s mode",
    async (viewMode) => {
      // Arrange: the workspace fileserver returns real .pptx bytes — a ZIP whose
      // header carries a NUL, so the hook classifies the file as binary.
      fetchMock.mockResolvedValue({
        ok: true,
        status: 200,
        arrayBuffer: () =>
          Promise.resolve(
            new Uint8Array([0x50, 0x4b, 0x03, 0x04, 0x00]).buffer,
          ),
      });

      // Act
      renderViewer("demo.pptx", viewMode);

      // Assert: the format-aware "no preview" message replaces the generic
      // binary fallback in both modes, so the pane is never blank.
      expect(
        await screen.findByTestId("file-content-viewer-unsupported-document"),
      ).toBeInTheDocument();
    },
  );

  // Regression test for the "PDF previews render blank — 'This page has been
  // blocked by Chrome'" bug (#16474). Chromium refuses to run its built-in
  // PDF viewer inside a sandboxed browsing context, so the PDF iframe must
  // NOT carry a `sandbox` attribute. The file is served same-origin from the
  // agent-server workspace fileserver, so omitting the sandbox keeps the
  // parent's origin rules intact while letting Chrome's PDF viewer engage.
  it("renders the PDF preview iframe WITHOUT a sandbox attribute so Chromium's built-in PDF viewer is not blocked", async () => {
    // Arrange: a `.pdf` extension routes the file through the `kind === "pdf"`
    // branch without fetching the body (image/pdf never go through `fetch`).
    renderViewer("deck.pdf", "rich");

    // Act: locate the preview iframe by its stable test id.
    const iframe = await screen.findByTestId("file-content-viewer-iframe");

    // Assert: the iframe points at the workspace fileserver URL (built from
    // the session base URL + relative path) — the base URL may carry a
    // cache-buster query param appended by the consumer, so we assert the
    // path prefix rather than the full string. Critically, it does NOT
    // carry a `sandbox` attribute, which is what caused "This page has
    // been blocked by Chrome" before the fix.
    expect(iframe.getAttribute("src")).toMatch(new RegExp(`^${BASE_URL.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}deck\\.pdf`));
    expect(iframe.hasAttribute("sandbox")).toBe(false);
  });
});
