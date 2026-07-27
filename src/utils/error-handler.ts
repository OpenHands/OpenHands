import { trackException } from "#/services/telemetry";

interface ErrorDetails {
  message: string;
  source?: string;
  metadata?: Record<string, unknown>;
}

export function trackError({ message, source, metadata = {} }: ErrorDetails) {
  const error = new Error(message);
  void trackException(error, {
    error_source: source || "unknown",
    ...metadata,
  });
}
