import { useTranslation } from "react-i18next";
import { useEventStore } from "#/stores/use-event-store";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";

/**
 * Rendered when the resume cursor had to step over `seq` intervals the
 * server skipped as unreadable. Those events cannot be refetched (the
 * server never resends them), so the conversation records the skip and the
 * chat says so instead of silently showing a hole in the transcript.
 * Cleared with the conversation's events.
 */
export function EventsGapNotice() {
  const skippedEventRanges = useEventStore((state) => state.skippedEventRanges);
  const { t } = useTranslation("openhands");

  if (skippedEventRanges.length === 0) return null;

  return (
    <div
      data-testid="events-gap-notice"
      className="mx-auto max-w-[90%] rounded-md border border-border bg-base-secondary px-3 py-2 text-center"
    >
      <Typography.Text className="text-xs text-muted">
        {t(I18nKey.CHAT_INTERFACE$EVENTS_SKIPPED_NOTICE)}
      </Typography.Text>
    </div>
  );
}
