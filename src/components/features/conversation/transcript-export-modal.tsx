import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import {
  BaseModalDescription,
  BaseModalTitle,
} from "#/components/shared/modals/confirmation-modals/base-modal";
import { BrandButton } from "#/components/features/settings/brand-button";
import { downloadBlob } from "#/utils/utils";
import {
  eventsToHtml,
  eventsToMarkdown,
  type TranscriptExportFormat,
  type TranscriptTruncation,
} from "#/utils/transcript-export";
import { useTracking } from "#/hooks/use-tracking";
import { useEventStore } from "#/stores/use-event-store";
import EventService from "#/api/event-service/event-service.api";
import {
  loadCompleteTranscriptEvents,
  loadBoundedTranscriptEvents,
  MAX_TRANSCRIPT_EXPORT_EVENTS,
  TRANSCRIPT_HEAD_EVENTS,
  TRANSCRIPT_TAIL_EVENTS,
} from "#/utils/transcript-export/load-complete-events";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

const TRANSCRIPT_FORMAT_RADIO_NAME = "transcript-export-format";
const TRANSCRIPT_SCOPE_RADIO_NAME = "transcript-export-scope";

/** Which slice of a very large conversation the user chose to download. */
type TranscriptExportScope = "partial" | "whole";

interface TranscriptExportModalProps {
  conversationId: string;
  conversationUrl?: string | null;
  sessionApiKey?: string | null;
  conversationTitle?: string | null;
  model?: string | null;
  onClose: () => void;
}

export function TranscriptExportModal({
  conversationId,
  conversationUrl,
  sessionApiKey,
  conversationTitle,
  model,
  onClose,
}: TranscriptExportModalProps) {
  const { t } = useTranslation("openhands");
  const { trackConversationExported } = useTracking();
  const [format, setFormat] =
    React.useState<TranscriptExportFormat>("markdown");
  const [includeToolDetails, setIncludeToolDetails] = React.useState(true);
  const [includeTimestamps, setIncludeTimestamps] = React.useState(true);
  const [scope, setScope] = React.useState<TranscriptExportScope>("partial");
  const [expectedEventCount, setExpectedEventCount] = React.useState<
    number | undefined
  >(undefined);
  const [isCheckingSize, setIsCheckingSize] = React.useState(true);
  const [isExporting, setIsExporting] = React.useState(false);
  const expectedEventCountRef = React.useRef<Promise<
    number | undefined
  > | null>(null);
  const isExportingRef = React.useRef(false);
  const isCancelledRef = React.useRef(false);
  const isMountedRef = React.useRef(true);

  React.useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      isCancelledRef.current = true;
    };
  }, []);

  React.useEffect(() => {
    // The count is used twice: it decides whether this conversation is large
    // enough to need the partial/whole choice, and the loaders use it to reject
    // silently truncated pagination. Archived cloud runtimes may no longer
    // expose this endpoint, while their persisted App API history remains
    // exportable, so treat the check as best-effort.
    const pendingEventCount = EventService.getEventCount(
      conversationId,
      conversationUrl ?? "",
      sessionApiKey,
    ).catch(() => undefined);
    expectedEventCountRef.current = pendingEventCount;

    let isCurrentRequest = true;
    pendingEventCount.then((count) => {
      if (!isCurrentRequest) return;
      setExpectedEventCount(count);
      setIsCheckingSize(false);
    });

    return () => {
      isCurrentRequest = false;
    };
  }, [conversationId, conversationUrl, sessionApiKey]);

  const isLargeConversation =
    expectedEventCount !== undefined &&
    expectedEventCount > MAX_TRANSCRIPT_EXPORT_EVENTS;

  const handleClose = () => {
    if (isExportingRef.current) isCancelledRef.current = true;
    onClose();
  };

  const handleExport = async () => {
    if (isExportingRef.current) return;

    isCancelledRef.current = false;
    isExportingRef.current = true;
    setIsExporting(true);
    try {
      const eventCount = await (expectedEventCountRef.current ??
        Promise.resolve(undefined));
      if (isCancelledRef.current) return;

      const eventStore = useEventStore.getState();
      const loadedEvents =
        eventStore.loadedConversationId === conversationId
          ? eventStore.events
          : [];
      const searchTranscriptEvents = (
        searchOptions: Parameters<typeof EventService.searchEvents>[3],
      ) =>
        EventService.searchEvents(
          conversationId,
          conversationUrl,
          sessionApiKey,
          searchOptions,
        );

      // For very large conversations the default is a bounded head+tail
      // window, so a 60k-event export never materializes in memory. Users who
      // need the omitted middle can opt into the whole history instead.
      // Smaller conversations always load in full.
      let events;
      let truncation: TranscriptTruncation | undefined;
      if (
        eventCount !== undefined &&
        eventCount > MAX_TRANSCRIPT_EXPORT_EVENTS &&
        scope === "partial"
      ) {
        const bounded = await loadBoundedTranscriptEvents(
          loadedEvents,
          searchTranscriptEvents,
          eventCount,
        );
        events = bounded.events;
        truncation = bounded.truncation;
      } else {
        events = await loadCompleteTranscriptEvents(
          loadedEvents,
          searchTranscriptEvents,
          eventCount,
        );
      }
      if (isCancelledRef.current) return;

      const options = {
        includeToolDetails,
        includeTimestamps,
        title: conversationTitle,
        model,
        truncation,
      };
      const isMarkdown = format === "markdown";
      const content = isMarkdown
        ? eventsToMarkdown(events, options)
        : eventsToHtml(events, options);
      const extension = isMarkdown ? "md" : "html";
      const mimeType = isMarkdown
        ? "text/markdown;charset=utf-8"
        : "text/html;charset=utf-8";

      downloadBlob(
        new Blob([content], { type: mimeType }),
        `conversation-${conversationId}.${extension}`,
      );
      trackConversationExported(format);
      onClose();
    } catch {
      if (!isCancelledRef.current) {
        displayErrorToast(t(I18nKey.ERROR$GENERIC));
      }
    } finally {
      isExportingRef.current = false;
      if (isMountedRef.current) setIsExporting(false);
    }
  };

  return (
    <ModalBackdrop
      onClose={handleClose}
      aria-label={t(I18nKey.TRANSCRIPT_EXPORT$TITLE)}
    >
      <ModalBody
        testID="transcript-export-modal"
        className="relative items-start border border-[var(--oh-border)]"
      >
        <ModalCloseButton
          onClose={handleClose}
          testId="close-transcript-export-modal"
          className="absolute right-4 top-4"
        />
        <div className="flex flex-col gap-2 pr-8">
          <BaseModalTitle title={t(I18nKey.TRANSCRIPT_EXPORT$TITLE)} />
          <BaseModalDescription
            description={t(I18nKey.TRANSCRIPT_EXPORT$DESCRIPTION)}
          />
        </div>

        <fieldset className="flex w-full flex-col gap-2">
          <legend className="mb-2 text-sm font-medium text-white">
            {t(I18nKey.TRANSCRIPT_EXPORT$FORMAT)}
          </legend>
          <label className="flex cursor-pointer items-center gap-3 rounded-md border border-[var(--oh-border)] px-3 py-2 text-sm text-white">
            <input
              type="radio"
              name={TRANSCRIPT_FORMAT_RADIO_NAME}
              value="markdown"
              disabled={isExporting}
              checked={format === "markdown"}
              onChange={() => setFormat("markdown")}
            />
            {t(I18nKey.TRANSCRIPT_EXPORT$MARKDOWN)}
          </label>
          <label className="flex cursor-pointer items-center gap-3 rounded-md border border-[var(--oh-border)] px-3 py-2 text-sm text-white">
            <input
              type="radio"
              name={TRANSCRIPT_FORMAT_RADIO_NAME}
              value="html"
              disabled={isExporting}
              checked={format === "html"}
              onChange={() => setFormat("html")}
            />
            {t(I18nKey.TRANSCRIPT_EXPORT$HTML)}
          </label>
        </fieldset>

        {isLargeConversation && (
          <fieldset
            className="flex w-full flex-col gap-2"
            data-testid="transcript-export-scope"
          >
            <legend className="mb-2 text-sm font-medium text-white">
              {t(I18nKey.TRANSCRIPT_EXPORT$SCOPE)}
            </legend>
            <p className="mb-1 text-xs text-modal-muted">
              {t(I18nKey.TRANSCRIPT_EXPORT$SCOPE_LARGE_NOTICE, {
                count: expectedEventCount,
              })}
            </p>
            <label className="flex cursor-pointer items-start gap-3 rounded-md border border-[var(--oh-border)] px-3 py-2 text-sm text-white">
              <input
                type="radio"
                className="mt-1"
                name={TRANSCRIPT_SCOPE_RADIO_NAME}
                value="partial"
                data-testid="transcript-export-scope-partial"
                disabled={isExporting}
                checked={scope === "partial"}
                onChange={() => setScope("partial")}
              />
              <span className="flex flex-col gap-1">
                {t(I18nKey.TRANSCRIPT_EXPORT$SCOPE_PARTIAL)}
                <span className="text-xs text-modal-muted">
                  {t(I18nKey.TRANSCRIPT_EXPORT$SCOPE_PARTIAL_HINT, {
                    head: TRANSCRIPT_HEAD_EVENTS,
                    tail: TRANSCRIPT_TAIL_EVENTS,
                  })}
                </span>
              </span>
            </label>
            <label className="flex cursor-pointer items-start gap-3 rounded-md border border-[var(--oh-border)] px-3 py-2 text-sm text-white">
              <input
                type="radio"
                className="mt-1"
                name={TRANSCRIPT_SCOPE_RADIO_NAME}
                value="whole"
                data-testid="transcript-export-scope-whole"
                disabled={isExporting}
                checked={scope === "whole"}
                onChange={() => setScope("whole")}
              />
              <span className="flex flex-col gap-1">
                {t(I18nKey.TRANSCRIPT_EXPORT$SCOPE_WHOLE)}
                <span className="text-xs text-warning">
                  {t(I18nKey.TRANSCRIPT_EXPORT$SCOPE_WHOLE_WARNING, {
                    count: expectedEventCount,
                  })}
                </span>
              </span>
            </label>
          </fieldset>
        )}

        <div className="flex w-full flex-col gap-3 text-sm text-white">
          <label className="flex cursor-pointer items-center gap-3">
            <input
              type="checkbox"
              disabled={isExporting}
              checked={includeToolDetails}
              onChange={(event) => setIncludeToolDetails(event.target.checked)}
            />
            {t(I18nKey.TRANSCRIPT_EXPORT$INCLUDE_TOOL_DETAILS)}
          </label>
          <label className="flex cursor-pointer items-center gap-3">
            <input
              type="checkbox"
              disabled={isExporting}
              checked={includeTimestamps}
              onChange={(event) => setIncludeTimestamps(event.target.checked)}
            />
            {t(I18nKey.TRANSCRIPT_EXPORT$INCLUDE_TIMESTAMPS)}
          </label>
        </div>

        <div className="flex w-full justify-end gap-2">
          <BrandButton
            type="button"
            variant="secondary"
            onClick={handleClose}
            testId="cancel-transcript-export"
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          <BrandButton
            type="button"
            variant="primary"
            onClick={handleExport}
            testId="confirm-transcript-export"
            isDisabled={isExporting || isCheckingSize}
            aria-busy={isExporting || isCheckingSize}
          >
            {t(I18nKey.BUTTON$EXPORT_CONVERSATION)}
          </BrandButton>
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
