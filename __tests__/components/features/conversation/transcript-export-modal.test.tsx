import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { TranscriptExportModal } from "#/components/features/conversation/transcript-export-modal";
import EventService from "#/api/event-service/event-service.api";
import {
  loadBoundedTranscriptEvents,
  loadCompleteTranscriptEvents,
  MAX_TRANSCRIPT_EXPORT_EVENTS,
} from "#/utils/transcript-export/load-complete-events";
import { downloadBlob } from "#/utils/utils";

vi.mock("#/utils/transcript-export/load-complete-events", async () => {
  const actual = await vi.importActual<
    typeof import("#/utils/transcript-export/load-complete-events")
  >("#/utils/transcript-export/load-complete-events");
  return {
    ...actual,
    loadCompleteTranscriptEvents: vi.fn(async () => []),
    loadBoundedTranscriptEvents: vi.fn(async () => ({
      events: [],
      truncation: { omittedCount: 42, headEventCount: 1 },
    })),
  };
});

vi.mock("#/utils/utils", async () => {
  const actual =
    await vi.importActual<typeof import("#/utils/utils")>("#/utils/utils");
  return { ...actual, downloadBlob: vi.fn() };
});

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({ trackConversationExported: vi.fn() }),
}));

const renderModal = (onClose = vi.fn()) =>
  renderWithProviders(
    <TranscriptExportModal
      conversationId="test-conversation-id"
      conversationUrl="http://localhost:3000"
      sessionApiKey="test-session-key"
      conversationTitle="Test Conversation"
      model="test-model"
      onClose={onClose}
    />,
  );

const exportButton = () => screen.getByTestId("confirm-transcript-export");

const waitForSizeCheck = async () =>
  waitFor(() => expect(exportButton()).toBeEnabled());

describe("TranscriptExportModal", () => {
  beforeEach(() => {
    vi.spyOn(EventService, "searchEvents").mockResolvedValue({
      items: [],
      next_page_id: null,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.restoreAllMocks();
  });

  describe("when the conversation exceeds the export threshold", () => {
    beforeEach(() => {
      vi.spyOn(EventService, "getEventCount").mockResolvedValue(
        MAX_TRANSCRIPT_EXPORT_EVENTS + 1,
      );
    });

    it("offers partial and whole downloads, with partial selected by default", async () => {
      renderModal();

      const partial = await screen.findByTestId(
        "transcript-export-scope-partial",
      );
      const whole = screen.getByTestId("transcript-export-scope-whole");

      expect(partial).toBeChecked();
      expect(whole).not.toBeChecked();
      // The i18n mock renders keys verbatim, so this asserts the size warning
      // is attached to the whole-conversation option.
      expect(
        screen.getByText("TRANSCRIPT_EXPORT$SCOPE_WHOLE_WARNING"),
      ).toBeInTheDocument();
    });

    it("exports the bounded head+tail window with the default choice", async () => {
      const onClose = vi.fn();
      renderModal(onClose);
      await waitForSizeCheck();

      await userEvent.click(exportButton());

      await waitFor(() => expect(downloadBlob).toHaveBeenCalled());
      expect(loadBoundedTranscriptEvents).toHaveBeenCalled();
      expect(loadCompleteTranscriptEvents).not.toHaveBeenCalled();
      expect(onClose).toHaveBeenCalled();
    });

    it("exports the entire history when the user chooses the whole conversation", async () => {
      const onClose = vi.fn();
      renderModal(onClose);
      await waitForSizeCheck();

      await userEvent.click(
        screen.getByTestId("transcript-export-scope-whole"),
      );
      await userEvent.click(exportButton());

      await waitFor(() => expect(downloadBlob).toHaveBeenCalled());
      expect(loadCompleteTranscriptEvents).toHaveBeenCalled();
      expect(loadBoundedTranscriptEvents).not.toHaveBeenCalled();
      expect(onClose).toHaveBeenCalled();
    });
  });

  describe("when the conversation is within the export threshold", () => {
    beforeEach(() => {
      vi.spyOn(EventService, "getEventCount").mockResolvedValue(
        MAX_TRANSCRIPT_EXPORT_EVENTS,
      );
    });

    it("exports everything in one click without offering a scope choice", async () => {
      renderModal();
      await waitForSizeCheck();

      expect(screen.queryByTestId("transcript-export-scope")).toBeNull();

      await userEvent.click(exportButton());

      await waitFor(() => expect(downloadBlob).toHaveBeenCalled());
      expect(loadCompleteTranscriptEvents).toHaveBeenCalled();
      expect(loadBoundedTranscriptEvents).not.toHaveBeenCalled();
    });
  });
});
