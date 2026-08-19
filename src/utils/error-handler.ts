import { trackEvent } from "#/services/telemetry";
import type { ErrorClassification } from "@openhands/typescript-client";

interface ErrorDetails {
  source?: string;
  metadata?: Record<string, unknown>;
  classification?: ErrorClassification | null;
}

const RESERVED_ERROR_KEYS = new Set([
  "error_source",
  "error_kind",
  "error_id",
  "error_telemetry",
]);

export function trackError({
  source,
  metadata = {},
  classification,
}: ErrorDetails) {
  // Reserved outcome fields are derived from `source`/`classification` and
  // must not be overridable through arbitrary caller metadata.
  const extra = Object.fromEntries(
    Object.entries(metadata).filter(([key]) => !RESERVED_ERROR_KEYS.has(key)),
  );
  const kind = classification?.kind || "unknown";
  // Promote a caller-provided `eventId` to the reserved `error_id` dimension as
  // a fallback so unclassified errors stay correlatable with server-side logs.
  const promotedEventId =
    typeof metadata.eventId === "string" ? metadata.eventId : undefined;
  if (promotedEventId != null) {
    delete extra.eventId;
  }
  const errorId = classification?.error_id ?? promotedEventId;

  void trackEvent("error_outcome", {
    ...extra,
    current_url: window.location.href,
    error_source: source || "unknown",
    error_kind: kind,
    // Keep diagnostic errors correlatable without capturing raw messages.
    ...(errorId != null ? { error_id: errorId } : {}),
    error_telemetry:
      kind === "internal" || kind === "unknown" ? "diagnostic" : "outcome",
  });
}
