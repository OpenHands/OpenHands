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

  void trackEvent("error_outcome", {
    ...extra,
    error_source: source || "unknown",
    error_kind: kind,
    // Keep diagnostic errors correlatable without capturing raw messages.
    ...(classification?.error_id != null
      ? { error_id: classification.error_id }
      : {}),
    error_telemetry:
      kind === "internal" || kind === "unknown"
        ? "diagnostic"
        : "outcome",
  });
}
