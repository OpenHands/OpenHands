export const FEATURE_DEV_PATH = "/feature-developer";
export const FEATURE_DEV_API_PATH = "/api/feature-developer/runs";
export const SESSION_API_KEY_HEADER = "X-Session-API-Key";
export const DEFAULT_MAX_CONCURRENT_AGENTS = 1;
export const FEATURE_DEV_ESTIMATE_USD_PER_CHAR = 0.00002;
export const FEATURE_DEV_MIN_PREVIEW_USD = 0.01;

export const FEATURE_DEV_STATUSES = [
  "pending",
  "running",
  "passed",
  "failed",
  "partial",
  "aborted",
  "paused",
] as const;

export const FEATURE_DEV_TICKET_STATUSES = [
  "pending",
  "in_progress",
  "passed",
  "failed",
  "skipped",
] as const;

export function featureDevRunPath(runId: string): string {
  return `${FEATURE_DEV_PATH}/${encodeURIComponent(runId)}`;
}

export function featureDevRunIdFromPath(currentPath: string): string | null {
  if (!currentPath.startsWith(`${FEATURE_DEV_PATH}/`)) {
    return null;
  }
  const rest = currentPath.slice(FEATURE_DEV_PATH.length + 1);
  if (!rest || rest.includes("/")) {
    return null;
  }
  return decodeURIComponent(rest);
}

export function previewEstimateUsd(spec: string): number {
  const length = spec.trim().length;
  if (!length) {
    return 0;
  }
  return Math.max(
    FEATURE_DEV_MIN_PREVIEW_USD,
    length * FEATURE_DEV_ESTIMATE_USD_PER_CHAR,
  );
}
