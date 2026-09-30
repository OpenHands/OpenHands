import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

// Mock the underlying workspace session / backend rather than the file-content
// hook, so the preview is exercised through the same URL-assembly code the app
// uses (the repository's testing rule).
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

import { OfficeArtifactPreview } from "#/components/features/chat/tool-visualizers/primitives/office-artifact-preview";
import { makeZip } from "../../../../../helpers/make-ooxml-zip";

const DOCX_XML = `<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
  <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Title</w:t></w:r></w:p>
  <w:p><w:r><w:t>Body text</w:t></w:r></w:p>
</w:body></w:document>`;

const BASE_URL =
  "https://agent.example.com/api/conversations/conv-1/workspace/";
const fetchMock = vi.fn();

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

async function stubZipFetch() {
  const bytes = await makeZip({ "word/document.xml": DOCX_XML });
  fetchMock.mockResolvedValue({
    ok: true,
    status: 200,
    arrayBuffer: async () => bytes,
    blob: async () => new Blob([bytes]),
  });
}

describe("OfficeArtifactPreview", () => {
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
  });

  it("unpacks a Word document fetched from the workspace fileserver", async () => {
    await stubZipFetch();

    render(<OfficeArtifactPreview path="notes.docx" />, {
      wrapper: makeWrapper(),
    });

    expect(
      await screen.findByText("Body text", undefined, { timeout: 3000 }),
    ).toBeInTheDocument();
    expect(screen.getByText("Title")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      `${BASE_URL}notes.docx`,
      expect.objectContaining({ credentials: "include" }),
    );
  });

  it("fetches the workspace-relative source path for an absolute event path", async () => {
    await stubZipFetch();

    render(
      <OfficeArtifactPreview
        path="/workspace/project/report.docx"
        sourcePath="report.docx"
      />,
      { wrapper: makeWrapper() },
    );

    await screen.findByText("Body text", undefined, { timeout: 3000 });
    expect(fetchMock).toHaveBeenCalledWith(
      `${BASE_URL}report.docx`,
      expect.objectContaining({ credentials: "include" }),
    );
    expect(fetchMock).not.toHaveBeenCalledWith(
      expect.stringContaining("/workspace/project/"),
      expect.anything(),
    );
  });

  it("shows the load error when the workspace fetch fails", async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 404 });

    render(<OfficeArtifactPreview path="notes.docx" />, {
      wrapper: makeWrapper(),
    });

    expect(
      await screen.findByTestId("office-artifact-preview-error"),
    ).toBeInTheDocument();
  });

  it("toggles the expanded height", async () => {
    await stubZipFetch();

    render(<OfficeArtifactPreview path="notes.docx" />, {
      wrapper: makeWrapper(),
    });
    const container = await screen.findByTestId(
      "office-artifact-preview-content",
      undefined,
      { timeout: 3000 },
    );

    expect(container).toHaveClass("max-h-48");
    await userEvent.click(screen.getByTestId("office-artifact-preview-expand"));
    await waitFor(() => expect(container).toHaveClass("max-h-[32rem]"));
  });

  it("renders the extracted outline as text content, not an opaque canvas", async () => {
    // The card's value to a model is the *text* it exposes: a vision model sees
    // the same words as pixels, and a text-only consumer reads them directly.
    await stubZipFetch();

    render(<OfficeArtifactPreview path="notes.docx" />, {
      wrapper: makeWrapper(),
    });

    const card = await screen.findByTestId(
      "office-artifact-preview",
      undefined,
      { timeout: 3000 },
    );
    await waitFor(() => expect(card).toHaveTextContent("Body text"));
    expect(card).toHaveTextContent("Title");
  });

  it("hides the View affordance when no handler is provided", async () => {
    await stubZipFetch();

    render(<OfficeArtifactPreview path="notes.docx" />, {
      wrapper: makeWrapper(),
    });
    await screen.findByTestId("office-artifact-preview-pending", undefined, {
      timeout: 3000,
    });

    expect(
      screen.queryByTestId("office-artifact-preview-view"),
    ).not.toBeInTheDocument();
  });

  it("offers Copy and Download like the other artifact cards", async () => {
    await stubZipFetch();

    render(<OfficeArtifactPreview path="notes.docx" content="source" />, {
      wrapper: makeWrapper(),
    });

    expect(
      await screen.findByTestId("office-artifact-preview-copy"),
    ).toBeInTheDocument();
    expect(
      await screen.findByTestId("office-artifact-preview-download"),
    ).toHaveAccessibleName(/download/i);
  });
});
