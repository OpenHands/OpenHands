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
    error_cause: classification?.cause || "unknown",
    error_impact: classification?.impact || "run_stopped",
    error_blame: classification?.blame || "unknown",
    error_telemetry: classification?.telemetry || "diagnostic",
    ...metadata,
  });
}
