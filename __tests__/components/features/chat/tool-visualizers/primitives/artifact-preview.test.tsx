import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

// The workspace *session* and *conversation* are the underlying services the
// preview reads through; mocking them (rather than `useWorkspaceFileContent`)
// keeps the real URL-assembly and backend-selection code under test, which is
// where the preview's bugs actually live.
const useWorkspaceSessionMock = vi.fn();
vi.mock("#/hooks/query/use-workspace-session", async (importOriginal) => {
  const real =
    await importOriginal<typeof import("#/hooks/query/use-workspace-session")>();
  return {
    ...real,
    useWorkspaceSession: () => useWorkspaceSessionMock(),
  };
});

const useActiveConversationMock = vi.fn();
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));

const useRuntimeIsReadyMock = vi.fn();
vi.mock("#/hooks/use-runtime-is-ready", () => ({
  useRuntimeIsReady: (...args: unknown[]) => useRuntimeIsReadyMock(...args),
}));

const getActiveBackendMock = vi.fn();
vi.mock("#/api/backend-registry/active-store", () => ({
  getActiveBackend: () => getActiveBackendMock(),
}));

const readCloudConversationFileMock = vi.fn();
vi.mock("#/api/cloud/conversation-service.api", () => ({
  readCloudConversationFile: (...args: unknown[]) =>
    readCloudConversationFileMock(...args),
}));

import { ArtifactPreview } from "#/components/features/chat/tool-visualizers/primitives/artifact-preview";

const fetchMock = vi.fn();
const BASE_URL =
  "https://agent.example.com/api/conversations/conv-1/workspace/";

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };
}

function renderPreview(ui: React.ReactElement) {
  return render(ui, { wrapper: makeWrapper() });
}

function textBytes(value: string): ArrayBuffer {
  return new TextEncoder().encode(value).buffer as ArrayBuffer;
}

describe("ArtifactPreview", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", fetchMock);
    fetchMock.mockReset();
    useWorkspaceSessionMock.mockReset();
    useActiveConversationMock.mockReset();
    useRuntimeIsReadyMock.mockReset();
    getActiveBackendMock.mockReset();
    readCloudConversationFileMock.mockReset();

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
  });

  it("renders the sandboxed frame from the workspace fileserver URL", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      arrayBuffer: () => Promise.resolve(textBytes("<h1>Hello</h1>")),
    });

    renderPreview(
      <ArtifactPreview path="index.html" content="<h1>Hello</h1>" />,
    );

    const frame = await screen.findByTestId("artifact-preview-frame");
    expect(frame).toHaveAttribute("sandbox", "allow-same-origin");
    // No script execution: agent-written HTML must stay inert.
    expect(frame.getAttribute("sandbox")).not.toContain("allow-scripts");
    expect(frame.getAttribute("src")).toContain(`${BASE_URL}index.html`);
  });

  it("fetches the workspace-relative source path, not the absolute event path", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      arrayBuffer: () => Promise.resolve(textBytes("<h1>Hello</h1>")),
    });

    renderPreview(
      <ArtifactPreview
        path="/workspace/project/report.html"
        sourcePath="report.html"
        content="<h1>Hello</h1>"
      />,
    );

    const frame = await screen.findByTestId("artifact-preview-frame");
    // The frame must point at the workspace-relative file, matching what View
    // opens; the absolute path would resolve under a duplicated root.
    expect(frame.getAttribute("src")).toContain(`${BASE_URL}report.html`);
    expect(frame.getAttribute("src")).not.toContain("/workspace/project/");
    // The display name still uses the original path's basename.
    expect(screen.getByText("report.html")).toBeInTheDocument();
  });

  it("does not cache-bust a Cloud data URL into an undecodable payload", async () => {
    getActiveBackendMock.mockReturnValue({
      backend: { id: "cloud-1", kind: "cloud" },
      orgId: null,
    });
    readCloudConversationFileMock.mockResolvedValue("<h1>Hello</h1>");

    renderPreview(
      <ArtifactPreview path="report.html" content="<h1>Hello</h1>" />,
    );

    const frame = await screen.findByTestId("artifact-preview-frame");
    const src = frame.getAttribute("src") ?? "";
    expect(src.startsWith("data:text/html")).toBe(true);
    // A `?v=` suffix on a data URL edits the base64 payload; the frame would
    // render nothing.
    expect(src).not.toContain("?v=");
  });

  it("copies the source text", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      arrayBuffer: () => Promise.resolve(textBytes("<h1>Hello</h1>")),
    });
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });

    renderPreview(
      <ArtifactPreview path="index.html" content="<h1>Hello</h1>" />,
    );
    await userEvent.click(await screen.findByTestId("artifact-preview-copy"));

    expect(writeText).toHaveBeenCalledWith("<h1>Hello</h1>");
  });

  it("downloads cross-origin files through a same-origin blob URL", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      arrayBuffer: () => Promise.resolve(textBytes("<h1>Hello</h1>")),
      blob: () => Promise.resolve(new Blob(["<h1>Hello</h1>"])),
    });
    const createObjectURL = vi.fn().mockReturnValue("blob:mock");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", {
      ...URL,
      createObjectURL,
      revokeObjectURL,
    });
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    try {
      renderPreview(
        <ArtifactPreview path="index.html" content="<h1>Hello</h1>" />,
      );
      await userEvent.click(
        await screen.findByTestId("artifact-preview-download"),
      );

      // The bytes are fetched (so the download is same-origin), then handed to
      // the browser as a blob URL — a cross-origin `download` attribute would
      // otherwise be ignored and the frame would navigate instead.
      await waitFor(() => expect(createObjectURL).toHaveBeenCalledTimes(1));
      expect(revokeObjectURL).toHaveBeenCalledWith("blob:mock");
      expect(click).toHaveBeenCalled();
    } finally {
      click.mockRestore();
      vi.unstubAllGlobals();
      vi.stubGlobal("fetch", fetchMock);
    }
  });

  it("gives the icon-only Download control an accessible name", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      arrayBuffer: () => Promise.resolve(textBytes("<h1>Hello</h1>")),
    });

    renderPreview(
      <ArtifactPreview path="index.html" content="<h1>Hello</h1>" />,
    );

    const button = await screen.findByTestId("artifact-preview-download");
    expect(button).toHaveAccessibleName(/download/i);
  });

  it("toggles the expanded height", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      arrayBuffer: () => Promise.resolve(textBytes("<h1>Hello</h1>")),
    });
    renderPreview(
      <ArtifactPreview path="index.html" content="<h1>Hello</h1>" />,
    );
    const container = screen.getByTestId("artifact-preview-frame-container");

    expect(container).toHaveClass("h-40");
    await userEvent.click(screen.getByTestId("artifact-preview-expand"));
    expect(container).toHaveClass("h-[32rem]");
  });

  it("hides the View affordance when no handler is provided", () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      arrayBuffer: () => Promise.resolve(textBytes("<h1>Hello</h1>")),
    });
    renderPreview(
      <ArtifactPreview path="index.html" content="<h1>Hello</h1>" />,
    );
    expect(
      screen.queryByTestId("artifact-preview-view"),
    ).not.toBeInTheDocument();
  });

  it("mounts the frame in view and unmounts it once scrolled away", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      arrayBuffer: () => Promise.resolve(textBytes("<h1>Hello</h1>")),
    });
    const observed: Array<(entries: { isIntersecting: boolean }[]) => void> =
      [];
    class MockIntersectionObserver {
      constructor(callback: (entries: { isIntersecting: boolean }[]) => void) {
        observed.push(callback);
      }
      observe() {}
      disconnect() {}
      unobserve() {}
    }
    vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);

    try {
      renderPreview(
        <ArtifactPreview path="index.html" content="<h1>Hello</h1>" />,
      );

      // Not yet visible → the iframe is not in the DOM.
      expect(
        screen.queryByTestId("artifact-preview-frame"),
      ).not.toBeInTheDocument();

      act(() => {
        observed.forEach((callback) => callback([{ isIntersecting: true }]));
      });
      expect(
        await screen.findByTestId("artifact-preview-frame"),
      ).toBeInTheDocument();

      // Scrolled far away → the live frame is dropped so a long conversation
      // does not accumulate one frame per visited card.
      act(() => {
        observed.forEach((callback) => callback([{ isIntersecting: false }]));
      });
      expect(
        screen.queryByTestId("artifact-preview-frame"),
      ).not.toBeInTheDocument();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("renders a raster image as an <img>, not a sandboxed frame", async () => {
    renderPreview(<ArtifactPreview path="logo.png" content="binary" />);

    const image = await screen.findByTestId("artifact-preview-image");
    expect(image.tagName).toBe("IMG");
    expect(image.getAttribute("src")).toContain(`${BASE_URL}logo.png`);
    expect(
      screen.queryByTestId("artifact-preview-frame"),
    ).not.toBeInTheDocument();
  });

  it("renders a PDF in an unsandboxed frame so the viewer can instantiate", async () => {
    renderPreview(<ArtifactPreview path="spec.pdf" content="binary" />);

    const frame = await screen.findByTestId("artifact-preview-pdf-frame");
    // Chromium will not instantiate the PDF plugin inside a sandboxed frame.
    expect(frame).not.toHaveAttribute("sandbox");
  });

  it("labels its content so the preview is machine-readable, not just pixels", async () => {
    // A multimodal model (e.g. DeepSeek V4.1 Flash) reads the rendered pixels;
    // a text-only consumer reads the DOM. Both need the element to be real
    // content with an accessible name, not a decorative box.
    const { unmount } = renderPreview(
      <ArtifactPreview path="logo.png" content="binary" />,
    );
    expect(
      await screen.findByTestId("artifact-preview-image"),
    ).toHaveAttribute("alt", "logo.png");
    unmount();

    renderPreview(<ArtifactPreview path="spec.pdf" content="binary" />);
    expect(
      await screen.findByTestId("artifact-preview-pdf-frame"),
    ).toHaveAttribute("title", "spec.pdf");
  });

  it("calls onView when provided", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      arrayBuffer: () => Promise.resolve(textBytes("<h1>Hello</h1>")),
    });
    const onView = vi.fn();
    renderPreview(
      <ArtifactPreview
        path="index.html"
        content="<h1>Hello</h1>"
        onView={onView}
      />,
    );

    await userEvent.click(await screen.findByTestId("artifact-preview-view"));
    expect(onView).toHaveBeenCalledTimes(1);
  });
});
