import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { TranscriptExportModal } from "#/components/features/conversation/transcript-export-modal";
import { I18nKey } from "#/i18n/declaration";
import type {
  HookExecutionEvent,
  OpenHandsEvent,
} from "#/types/agent-server/core";

const {
  displayErrorToastMock,
  downloadBlobMock,
  eventStoreState,
  eventsToHtmlMock,
  eventsToMarkdownMock,
  getEventCountMock,
  loadCompleteTranscriptEventsMock,
  loadBoundedTranscriptEventsMock,
  searchEventsMock,
  trackConversationExportedMock,
  useTranslationMock,
} = vi.hoisted(() => ({
  displayErrorToastMock: vi.fn(),
  downloadBlobMock: vi.fn(),
  eventStoreState: {
    loadedConversationId: null as string | null,
    events: [] as OpenHandsEvent[],
  },
  eventsToHtmlMock: vi.fn(),
  eventsToMarkdownMock: vi.fn(),
  getEventCountMock: vi.fn(),
  loadCompleteTranscriptEventsMock: vi.fn(),
  loadBoundedTranscriptEventsMock: vi.fn(),
  searchEventsMock: vi.fn(),
  trackConversationExportedMock: vi.fn(),
  useTranslationMock: vi.fn(),
}));

vi.mock("react-i18next", async (importOriginal) => ({
  ...(await importOriginal<typeof import("react-i18next")>()),
  useTranslation: (namespace: string) => {
    useTranslationMock(namespace);
    return { t: (key: string) => key };
  },
}));

vi.mock("#/utils/utils", async (importOriginal) => ({
  ...(await importOriginal<typeof import("#/utils/utils")>()),
  downloadBlob: (...args: unknown[]) => downloadBlobMock(...args),
}));

vi.mock("#/utils/transcript-export", () => ({
  eventsToHtml: (...args: unknown[]) => eventsToHtmlMock(...args),
  eventsToMarkdown: (...args: unknown[]) => eventsToMarkdownMock(...args),
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackConversationExported: (...args: unknown[]) =>
      trackConversationExportedMock(...args),
  }),
}));

vi.mock("#/stores/use-event-store", () => ({
  useEventStore: {
    getState: () => eventStoreState,
  },
}));

vi.mock("#/api/event-service/event-service.api", () => ({
  default: {
    getEventCount: (...args: unknown[]) => getEventCountMock(...args),
    searchEvents: (...args: unknown[]) => searchEventsMock(...args),
  },
}));

vi.mock("#/utils/transcript-export/load-complete-events", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("#/utils/transcript-export/load-complete-events")
  >();
  return {
    ...actual,
    loadCompleteTranscriptEvents: (...args: unknown[]) =>
      loadCompleteTranscriptEventsMock(...args),
    loadBoundedTranscriptEvents: (...args: unknown[]) =>
      loadBoundedTranscriptEventsMock(...args),
  };
});

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: (...args: unknown[]) => displayErrorToastMock(...args),
}));

const makeEvent = (id: string): HookExecutionEvent => ({
  id,
  timestamp: "2026-07-12T00:00:00.000Z",
  source: "hook",
  kind: "HookExecutionEvent",
  hook_event_type: "PreToolUse",
  hook_command: "npm test",
  success: true,
  blocked: false,
  exit_code: 0,
  reason: null,
  tool_name: "terminal",
  action_id: null,
  message_id: null,
  stdout: null,
  stderr: null,
  error: null,
  additional_context: null,
  hook_input: null,
});

function createDeferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

async function readBlob(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(blob);
  });
}

interface RenderExportModalOptions {
  loadedConversationId?: string | null;
  loadedEvents?: OpenHandsEvent[];
  conversationUrl?: string | null;
  sessionApiKey?: string | null;
  conversationTitle?: string | null;
  model?: string | null;
  eventCount?: number;
  countUnavailable?: boolean;
}

function renderExportModal({
  loadedConversationId = "conversation-1",
  loadedEvents = [makeEvent("loaded-event")],
  conversationUrl,
  sessionApiKey = null,
  conversationTitle = null,
  model = null,
  eventCount = 1,
  countUnavailable = false,
}: RenderExportModalOptions = {}) {
  [
    displayErrorToastMock,
    downloadBlobMock,
    eventsToHtmlMock,
    eventsToMarkdownMock,
    getEventCountMock,
    loadCompleteTranscriptEventsMock,
    loadBoundedTranscriptEventsMock,
    searchEventsMock,
    trackConversationExportedMock,
    useTranslationMock,
  ].forEach((mock) => mock.mockReset());

  eventStoreState.loadedConversationId = loadedConversationId;
  eventStoreState.events = loadedEvents;
  const completeEvents = [makeEvent("complete-event")];
  if (countUnavailable) {
    getEventCountMock.mockRejectedValue(new Error("count unavailable"));
  } else {
    getEventCountMock.mockResolvedValue(eventCount);
  }
  searchEventsMock.mockResolvedValue({ items: [], next_page_id: null });
  loadCompleteTranscriptEventsMock.mockResolvedValue(completeEvents);
  loadBoundedTranscriptEventsMock.mockResolvedValue({
    events: completeEvents,
    truncation: { omittedCount: 42, headEventCount: 1 },
  });
  eventsToMarkdownMock.mockReturnValue("# Markdown transcript");
  eventsToHtmlMock.mockReturnValue("<html>HTML transcript</html>");
  const onClose = vi.fn();
  const view = renderWithProviders(
    <TranscriptExportModal
      conversationId="conversation-1"
      conversationUrl={conversationUrl}
      sessionApiKey={sessionApiKey}
      conversationTitle={conversationTitle}
      model={model}
      onClose={onClose}
    />,
  );

  return { ...view, completeEvents, loadedEvents, onClose };
}

describe("transcript export modal", () => {
  it("exports the current transcript as Markdown with accessible defaults", async () => {
    const scenario = renderExportModal();
    const markdownRadio = screen.getByRole("radio", {
      name: I18nKey.TRANSCRIPT_EXPORT$MARKDOWN,
    });
    const htmlRadio = screen.getByRole("radio", {
      name: I18nKey.TRANSCRIPT_EXPORT$HTML,
    });

    expect(markdownRadio).toBeChecked();
    expect(htmlRadio).not.toBeChecked();
    expect(markdownRadio).toHaveAttribute("name", "transcript-export-format");
    expect(htmlRadio).toHaveAttribute("name", "transcript-export-format");
    expect(
      screen.getByRole("checkbox", {
        name: I18nKey.TRANSCRIPT_EXPORT$INCLUDE_TOOL_DETAILS,
      }),
    ).toBeChecked();
    expect(
      screen.getByRole("checkbox", {
        name: I18nKey.TRANSCRIPT_EXPORT$INCLUDE_TIMESTAMPS,
      }),
    ).toBeChecked();
    expect(useTranslationMock).toHaveBeenNthCalledWith(1, "openhands");

    // The export button stays disabled until the on-mount size check resolves.
    await waitFor(() =>
      expect(screen.getByTestId("confirm-transcript-export")).toBeEnabled(),
    );
    fireEvent.click(screen.getByTestId("confirm-transcript-export"));

    await waitFor(() => expect(downloadBlobMock).toHaveBeenCalledOnce());
    expect(getEventCountMock).toHaveBeenCalledWith("conversation-1", "", null);
    expect(loadBoundedTranscriptEventsMock).not.toHaveBeenCalled();
    expect(loadCompleteTranscriptEventsMock).toHaveBeenCalledWith(
      scenario.loadedEvents,
      expect.any(Function),
      1,
    );
    expect(eventsToMarkdownMock).toHaveBeenCalledWith(scenario.completeEvents, {
      includeToolDetails: true,
      includeTimestamps: true,
      title: null,
      model: null,
      truncation: undefined,
    });
    expect(eventsToHtmlMock).not.toHaveBeenCalled();
    const [blob, filename] = downloadBlobMock.mock.calls[0] as [Blob, string];
    expect(filename).toBe("conversation-conversation-1.md");
    expect(blob.type).toBe("text/markdown;charset=utf-8");
    await expect(readBlob(blob)).resolves.toBe("# Markdown transcript");
    expect(trackConversationExportedMock).toHaveBeenCalledWith("markdown");
    expect(scenario.onClose).toHaveBeenCalledOnce();
  });

  it("exports fetched events as HTML with the selected options", async () => {
    const scenario = renderExportModal({
      loadedConversationId: "another-conversation",
      conversationUrl: "wss://runtime.example/ws",
      sessionApiKey: "session-key",
      conversationTitle: "Incident review",
      model: "model-v2",
    });
    const searchOptions = {
      limit: 100,
      sortOrder: "TIMESTAMP_DESC",
      strictPagination: true,
    };
    loadCompleteTranscriptEventsMock.mockImplementation(
      async (
        _loadedEvents: OpenHandsEvent[],
        searchEvents: (options: Record<string, unknown>) => Promise<unknown>,
      ) => {
        await searchEvents(searchOptions);
        return scenario.completeEvents;
      },
    );
    const markdownRadio = screen.getByRole("radio", {
      name: I18nKey.TRANSCRIPT_EXPORT$MARKDOWN,
    });
    const htmlRadio = screen.getByRole("radio", {
      name: I18nKey.TRANSCRIPT_EXPORT$HTML,
    });

    fireEvent.click(htmlRadio);
    expect(htmlRadio).toBeChecked();
    fireEvent.click(markdownRadio);
    expect(markdownRadio).toBeChecked();
    fireEvent.click(htmlRadio);
    fireEvent.click(
      screen.getByRole("checkbox", {
        name: I18nKey.TRANSCRIPT_EXPORT$INCLUDE_TOOL_DETAILS,
      }),
    );
    fireEvent.click(
      screen.getByRole("checkbox", {
        name: I18nKey.TRANSCRIPT_EXPORT$INCLUDE_TIMESTAMPS,
      }),
    );
    await waitFor(() =>
      expect(screen.getByTestId("confirm-transcript-export")).toBeEnabled(),
    );
    fireEvent.click(screen.getByTestId("confirm-transcript-export"));

    await waitFor(() => expect(downloadBlobMock).toHaveBeenCalledOnce());
    expect(loadCompleteTranscriptEventsMock).toHaveBeenCalledWith(
      [],
      expect.any(Function),
      1,
    );
    expect(searchEventsMock).toHaveBeenCalledWith(
      "conversation-1",
      "wss://runtime.example/ws",
      "session-key",
      searchOptions,
    );
    expect(eventsToHtmlMock).toHaveBeenCalledWith(scenario.completeEvents, {
      includeToolDetails: false,
      includeTimestamps: false,
      title: "Incident review",
      model: "model-v2",
      truncation: undefined,
    });
    expect(eventsToMarkdownMock).not.toHaveBeenCalled();
    const [blob, filename] = downloadBlobMock.mock.calls[0] as [Blob, string];
    expect(filename).toBe("conversation-conversation-1.html");
    expect(blob.type).toBe("text/html;charset=utf-8");
    await expect(readBlob(blob)).resolves.toBe("<html>HTML transcript</html>");
    expect(trackConversationExportedMock).toHaveBeenCalledWith("html");
    expect(scenario.onClose).toHaveBeenCalledOnce();
  });

  it("continues without an expected count when that endpoint is unavailable", async () => {
    const scenario = renderExportModal({ countUnavailable: true });

    await waitFor(() =>
      expect(screen.getByTestId("confirm-transcript-export")).toBeEnabled(),
    );
    fireEvent.click(screen.getByTestId("confirm-transcript-export"));

    await waitFor(() => expect(downloadBlobMock).toHaveBeenCalledOnce());
    expect(loadCompleteTranscriptEventsMock).toHaveBeenCalledWith(
      scenario.loadedEvents,
      expect.any(Function),
      undefined,
    );
    expect(loadBoundedTranscriptEventsMock).not.toHaveBeenCalled();
    expect(displayErrorToastMock).not.toHaveBeenCalled();
  });

  it("offers partial and whole downloads, with partial selected by default, for a large conversation", async () => {
    renderExportModal({ eventCount: 10_001 });

    const partial = await screen.findByTestId(
      "transcript-export-scope-partial",
    );
    const whole = screen.getByTestId("transcript-export-scope-whole");

    expect(partial).toBeChecked();
    expect(whole).not.toBeChecked();
    expect(
      screen.getByText("TRANSCRIPT_EXPORT$SCOPE_WHOLE_WARNING"),
    ).toBeInTheDocument();
  });

  it("exports the bounded head+tail window by default for a large conversation", async () => {
    const scenario = renderExportModal({ eventCount: 10_001 });

    await screen.findByTestId("transcript-export-scope");
    fireEvent.click(screen.getByTestId("confirm-transcript-export"));

    await waitFor(() => expect(downloadBlobMock).toHaveBeenCalledOnce());
    expect(loadBoundedTranscriptEventsMock).toHaveBeenCalledWith(
      scenario.loadedEvents,
      expect.any(Function),
      10_001,
    );
    expect(loadCompleteTranscriptEventsMock).not.toHaveBeenCalled();
    expect(eventsToMarkdownMock).toHaveBeenCalledWith(scenario.completeEvents, {
      includeToolDetails: true,
      includeTimestamps: true,
      title: null,
      model: null,
      truncation: { omittedCount: 42, headEventCount: 1 },
    });
    expect(scenario.onClose).toHaveBeenCalledOnce();
  });

  it("exports the entire history when the user chooses the whole conversation", async () => {
    const scenario = renderExportModal({ eventCount: 10_001 });

    await screen.findByTestId("transcript-export-scope");
    await userEvent.click(screen.getByTestId("transcript-export-scope-whole"));
    await userEvent.click(screen.getByTestId("confirm-transcript-export"));

    await waitFor(() => expect(downloadBlobMock).toHaveBeenCalledOnce());
    expect(loadCompleteTranscriptEventsMock).toHaveBeenCalledWith(
      scenario.loadedEvents,
      expect.any(Function),
      10_001,
    );
    expect(loadBoundedTranscriptEventsMock).not.toHaveBeenCalled();
    expect(scenario.onClose).toHaveBeenCalledOnce();
  });

  it("exports everything without a scope choice when the conversation is within the threshold", async () => {
    const scenario = renderExportModal({ eventCount: 10_000 });

    await waitFor(() =>
      expect(screen.getByTestId("confirm-transcript-export")).toBeEnabled(),
    );
    expect(screen.queryByTestId("transcript-export-scope")).toBeNull();

    fireEvent.click(screen.getByTestId("confirm-transcript-export"));

    await waitFor(() => expect(downloadBlobMock).toHaveBeenCalledOnce());
    expect(loadCompleteTranscriptEventsMock).toHaveBeenCalled();
    expect(loadBoundedTranscriptEventsMock).not.toHaveBeenCalled();
    expect(scenario.onClose).toHaveBeenCalledOnce();
  });

  it("reports an export error, restores the form, and permits a retry", async () => {
    const scenario = renderExportModal();
    loadCompleteTranscriptEventsMock.mockRejectedValueOnce(
      new Error("history unavailable"),
    );
    const exportButton = screen.getByTestId("confirm-transcript-export");

    await waitFor(() => expect(exportButton).toBeEnabled());
    fireEvent.click(exportButton);

    await waitFor(() =>
      expect(displayErrorToastMock).toHaveBeenCalledWith(I18nKey.ERROR$GENERIC),
    );
    expect(downloadBlobMock).not.toHaveBeenCalled();
    expect(scenario.onClose).not.toHaveBeenCalled();
    await waitFor(() => expect(exportButton).not.toBeDisabled());
    expect(exportButton).toHaveAttribute("aria-busy", "false");

    fireEvent.click(exportButton);

    await waitFor(() => expect(downloadBlobMock).toHaveBeenCalledOnce());
    expect(loadCompleteTranscriptEventsMock).toHaveBeenCalledTimes(2);
    expect(trackConversationExportedMock).toHaveBeenCalledWith("markdown");
    expect(scenario.onClose).toHaveBeenCalledOnce();
  });

  it("cancels after history loading without creating a download", async () => {
    const scenario = renderExportModal();
    const history = createDeferred<OpenHandsEvent[]>();
    loadCompleteTranscriptEventsMock.mockReturnValue(history.promise);

    await waitFor(() =>
      expect(screen.getByTestId("confirm-transcript-export")).toBeEnabled(),
    );
    fireEvent.click(screen.getByTestId("confirm-transcript-export"));
    await waitFor(() =>
      expect(loadCompleteTranscriptEventsMock).toHaveBeenCalledOnce(),
    );
    fireEvent.click(screen.getByTestId("close-transcript-export-modal"));
    await act(async () => history.resolve(scenario.completeEvents));

    expect(scenario.onClose).toHaveBeenCalledOnce();
    expect(eventsToMarkdownMock).not.toHaveBeenCalled();
    expect(downloadBlobMock).not.toHaveBeenCalled();
    expect(displayErrorToastMock).not.toHaveBeenCalled();
  });

  it("suppresses late errors and state updates after unmount", async () => {
    const scenario = renderExportModal();
    const history = createDeferred<OpenHandsEvent[]>();
    loadCompleteTranscriptEventsMock.mockReturnValue(history.promise);

    await waitFor(() =>
      expect(screen.getByTestId("confirm-transcript-export")).toBeEnabled(),
    );
    fireEvent.click(screen.getByTestId("confirm-transcript-export"));
    await waitFor(() =>
      expect(loadCompleteTranscriptEventsMock).toHaveBeenCalledOnce(),
    );
    scenario.unmount();
    await act(async () => history.reject(new Error("late failure")));

    expect(displayErrorToastMock).not.toHaveBeenCalled();
    expect(downloadBlobMock).not.toHaveBeenCalled();
    expect(scenario.onClose).not.toHaveBeenCalled();
  });

  it("ignores a second export request while the first is running", async () => {
    const scenario = renderExportModal();
    const history = createDeferred<OpenHandsEvent[]>();
    loadCompleteTranscriptEventsMock.mockReturnValue(history.promise);
    const exportButton = screen.getByTestId("confirm-transcript-export");

    await waitFor(() => expect(exportButton).toBeEnabled());
    act(() => {
      exportButton.click();
      exportButton.click();
    });
    await act(async () => history.resolve(scenario.completeEvents));

    await waitFor(() => expect(downloadBlobMock).toHaveBeenCalledOnce());
    expect(loadCompleteTranscriptEventsMock).toHaveBeenCalledOnce();
    expect(trackConversationExportedMock).toHaveBeenCalledOnce();
    expect(scenario.onClose).toHaveBeenCalledOnce();
  });

  it("closes immediately without starting an export", () => {
    const scenario = renderExportModal();

    fireEvent.click(screen.getByTestId("close-transcript-export-modal"));

    expect(scenario.onClose).toHaveBeenCalledOnce();
    expect(loadCompleteTranscriptEventsMock).not.toHaveBeenCalled();
    expect(loadBoundedTranscriptEventsMock).not.toHaveBeenCalled();
  });
});