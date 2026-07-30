import { trackEvent } from "#/services/telemetry";
import { ErrorClassification } from "#/types/error-classification";

interface ErrorDetails {
  message: string;
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
    error_source: source || "unknown",
    error_kind: classification?.kind || "unknown",
    error_telemetry:
      classification?.kind === "internal" ? "diagnostic" : "outcome",
    ...metadata,
  });
}
