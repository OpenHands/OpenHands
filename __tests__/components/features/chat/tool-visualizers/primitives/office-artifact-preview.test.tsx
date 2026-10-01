import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

// Mock the underlying workspace session / backend rather than the file-content
// hook, so the preview is exercised through the same URL-assembly code the app
// uses (the repository's testing rule).
const useWorkspaceSessionMock = vi.fn();
vi.mock("#/hooks/query/use-workspace-session", async (importOriginal) => {
  const real =
    await importOriginal<
      typeof import("#/hooks/query/use-workspace-session")
    >();
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

import { OfficeArtifactPreview } from "#/components/features/chat/tool-visualizers/primitives/office-artifact-preview";
import { MAX_OOXML_DOWNLOAD_BYTES } from "#/utils/ooxml-preview";
import { I18nKey } from "#/i18n/declaration";
import { useWorkspaceMutationCounter } from "#/stores/use-workspace-mutation-counter";
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
    // The mutation counter is module-global; reset it so a bump in one test
    // cannot leave the next test's bytes tagged with a stale version.
    useWorkspaceMutationCounter.setState({ count: 0 });
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
      `${BASE_URL}notes.docx?v=0`,
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
      `${BASE_URL}report.docx?v=0`,
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

  it("refuses an oversized document before buffering it", async () => {
    // The reader caps *unpacked* parts, but the container itself is downloaded
    // whole — a declared length past the download cap must stop the body from
    // being buffered at all (the file-content hook owns that bound now).
    const cancel = vi.fn().mockResolvedValue(undefined);
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      headers: new Headers({
        "content-length": String(MAX_OOXML_DOWNLOAD_BYTES + 1),
      }),
      body: { cancel },
      arrayBuffer: async () => new ArrayBuffer(0),
      blob: async () => new Blob([]),
    });

    render(<OfficeArtifactPreview path="huge.docx" />, {
      wrapper: makeWrapper(),
    });

    // Translations are not loaded in the test env, so the rendered text is the
    // i18n key: asserting the size-specific key proves the distinction, where
    // the generic load-error key would mean the rejection was misclassified.
    expect(
      await screen.findByTestId("office-artifact-preview-error", undefined, {
        timeout: 3000,
      }),
    ).toHaveTextContent(I18nKey.FILES$FILE_TOO_LARGE);
    // Rejecting on the declared length happens before the body is read, so the
    // stream must be cancelled or the browser keeps downloading a file the
    // preview will never parse.
    expect(cancel).toHaveBeenCalled();
  });

  it("re-parses when the cache-busted URL changes", async () => {
    // An agent-side rewrite bumps the mutation counter, which changes the URL.
    // The outline must follow it: skipping the re-parse once a result exists
    // leaves the card showing the pre-edit document (or a stale 404).
    const first = await makeZip({ "word/document.xml": DOCX_XML });
    const second = await makeZip({
      "word/document.xml": DOCX_XML.replace("Body text", "Edited text"),
    });
    // The file-content hook is the only fetcher now: it classifies the file
    // (initial request) and refetches after the mutation bump, so the second
    // response is the edited document the outline must switch to.
    fetchMock
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        arrayBuffer: async () => first,
        blob: async () => new Blob([first]),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        arrayBuffer: async () => second,
        blob: async () => new Blob([second]),
      });

    render(<OfficeArtifactPreview path="notes.docx" />, {
      wrapper: makeWrapper(),
    });
    await screen.findByText("Body text", undefined, { timeout: 3000 });

    act(() => useWorkspaceMutationCounter.getState().bump());

    expect(
      await screen.findByText("Edited text", undefined, { timeout: 3000 }),
    ).toBeInTheDocument();
    expect(screen.queryByText("Body text")).not.toBeInTheDocument();
  });

  it("never shows pre-edit bytes as the edited document's version", async () => {
    // The hook refetches under a new counter after an edit, and the card must
    // not label the previous version's bytes with the new URL. With the refetch
    // still in flight the card shows its pending state — never the stale
    // outline dressed up as the edited document.
    const stale = await makeZip({ "word/document.xml": DOCX_XML });
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      arrayBuffer: async () => stale,
      blob: async () => new Blob([stale]),
    });

    render(<OfficeArtifactPreview path="notes.docx" />, {
      wrapper: makeWrapper(),
    });
    await screen.findByText("Body text", undefined, { timeout: 3000 });

    // Hold the refetch open so the version-0 bytes are all the hook can offer.
    fetchMock.mockReturnValue(new Promise(() => {}));
    act(() => useWorkspaceMutationCounter.getState().bump());

    expect(
      await screen.findByTestId("office-artifact-preview-pending"),
    ).toBeInTheDocument();
    expect(screen.queryByText("Body text")).not.toBeInTheDocument();
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

  it("does not run the outline on cloud, where binary bytes are lossy", async () => {
    // The Cloud file API returns content as a UTF-8-decoded string, so a ZIP
    // container's bytes are already destroyed. The card must withhold the
    // outline (rather than show a parse failure) and still offer Download.
    getActiveBackendMock.mockReturnValue({
      backend: { id: "cloud-1", kind: "cloud" },
      orgId: null,
    });
    // A string containing a NUL survives the hook's binary sniff, which is the
    // path that produces the lossy-bytes result.
    readCloudConversationFileMock.mockResolvedValue("PK\u0000lossy");

    render(<OfficeArtifactPreview path="notes.docx" />, {
      wrapper: makeWrapper(),
    });

    expect(
      await screen.findByTestId("office-artifact-preview-error", undefined, {
        timeout: 3000,
      }),
    ).toHaveTextContent(I18nKey.FILES$BINARY_FALLBACK);
    expect(
      screen.getByTestId("office-artifact-preview-download"),
    ).toBeInTheDocument();
    // No byte-accurate fetch is attempted on this path.
    expect(fetchMock).not.toHaveBeenCalled();
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
