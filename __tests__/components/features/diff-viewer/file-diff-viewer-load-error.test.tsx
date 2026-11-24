import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { server } from "#/mocks/node";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { FileDiffViewer } from "#/components/features/diff-viewer/file-diff-viewer";

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "conv-1" }),
}));
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      conversation_url: "https://runtime.example.test/api/conversations/conv-1",
      session_api_key: "session-key",
      workspace: { working_dir: "/workspace/project" },
    },
  }),
}));
// Monaco cannot run in jsdom.
vi.mock("@monaco-editor/react", () => ({
  DiffEditor: (props: { modified: string }) => (
    <div data-testid="file-diff-viewer" data-modified={props.modified} />
  ),
  Editor: () => <div data-testid="file-single-viewer" />,
}));

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Cloud",
  host: "https://app.example.test",
  apiKey: "cloud-key",
  kind: "cloud",
};
const DIFF_URL = `${cloudBackend.host}/api/v1/app-conversations/conv-1/git/diff`;

const failure = () =>
  HttpResponse.json({ detail: "unavailable" }, { status: 502 });

function renderViewer() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={client}>
      <FileDiffViewer path="src/app.ts" type="M" />
    </QueryClientProvider>,
  );
  return client;
}

describe("FileDiffViewer diff load failures", () => {
  beforeEach(() => {
    __resetActiveStoreForTests();
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-1" });
  });

  afterEach(() => {
    __resetActiveStoreForTests();
  });

  it("shows an inline error with Retry, and Retry loads the diff", async () => {
    // Arrange
    let fail = true;
    server.use(
      http.get(DIFF_URL, () =>
        fail
          ? failure()
          : HttpResponse.json({ original: "old", modified: "new" }),
      ),
    );
    const user = userEvent.setup();
    renderViewer();

    // Act
    await user.click(screen.getByTestId("collapse"));

    // Assert
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("DIFF_VIEWER$LOAD_FAILED");
    expect(screen.queryByTestId("file-diff-viewer")).not.toBeInTheDocument();

    // Act
    fail = false;
    await user.click(screen.getByRole("button", { name: "DIFF_VIEWER$RETRY" }));

    // Assert
    expect(await screen.findByTestId("file-diff-viewer")).toHaveAttribute(
      "data-modified",
      "new",
    );
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("keeps the last loaded diff when a refresh fails", async () => {
    // Arrange
    let fail = false;
    server.use(
      http.get(DIFF_URL, () =>
        fail ? failure() : HttpResponse.json({ original: "a", modified: "b" }),
      ),
    );
    const user = userEvent.setup();
    const client = renderViewer();
    await user.click(screen.getByTestId("collapse"));
    await screen.findByTestId("file-diff-viewer");

    // Act
    fail = true;
    await client.invalidateQueries({ queryKey: ["file_diff"] });

    // Assert
    await waitFor(() =>
      expect(
        client.getQueryCache().findAll({ queryKey: ["file_diff"] })[0].state
          .status,
      ).toBe("error"),
    );
    expect(screen.getByTestId("file-diff-viewer")).toHaveAttribute(
      "data-modified",
      "b",
    );
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("treats an empty successful diff as loaded", async () => {
    // Arrange
    server.use(
      http.get(DIFF_URL, () =>
        HttpResponse.json({ original: "", modified: "" }),
      ),
    );
    const user = userEvent.setup();
    renderViewer();

    // Act
    await user.click(screen.getByTestId("collapse"));

    // Assert
    expect(await screen.findByTestId("file-diff-viewer")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
