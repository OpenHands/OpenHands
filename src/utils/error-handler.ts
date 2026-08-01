import { trackEvent } from "#/services/telemetry";
import type { ErrorClassification } from "@openhands/typescript-client";

interface ErrorDetails {
  source?: string;
  metadata?: Record<string, unknown>;
  classification?: ErrorClassification | null;
}

export function trackError({
  source,
  metadata = {},
  classification,
}: ErrorDetails) {
  void trackEvent("error_outcome", {
    ...metadata,
    error_source: source || "unknown",
    error_kind: classification?.kind || "unknown",
    // Keep diagnostic errors correlatable without capturing raw messages.
    ...(classification?.error_id != null
      ? { error_id: classification.error_id }
      : {}),
    error_telemetry:
      classification == null ||
      classification.kind === "internal" ||
      classification.kind === "unknown"
        ? "diagnostic"
        : "outcome",
  });
}
