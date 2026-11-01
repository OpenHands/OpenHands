/**
 * Structured plugin error payload from the agent-server.
 *
 * When a plugin manifest fails to load, the SDK raises `PluginManifestError`
 * which the agent-server catches and returns as HTTP 400 with this shape
 * in the response `detail` field.
 */
export interface PluginErrorDetail {
  plugin_name: string;
  failure_reason: string;
  remediation: string;
  error_id: string;
}

/**
 * Type guard: does this error response detail look like a PluginManifestError?
 *
 * The agent-server returns `detail` as a dict with these exact keys when
 * a plugin fails to load. We detect it by checking for the presence of
 * `plugin_name` and `error_id` (the two fields unique to plugin errors).
 */
export function isPluginErrorDetail(
  detail: unknown,
): detail is PluginErrorDetail {
  if (!detail || typeof detail !== "object") return false;
  const d = detail as Record<string, unknown>;
  return (
    typeof d.plugin_name === "string" &&
    typeof d.failure_reason === "string" &&
    typeof d.remediation === "string" &&
    typeof d.error_id === "string"
  );
}

/**
 * Extract a PluginErrorDetail from an axios error response.
 *
 * The agent-server returns HTTP 400 with `{ detail: { plugin_name, ... } }`.
 * This function safely extracts that detail from various error shapes.
 */
export function extractPluginErrorDetail(
  error: unknown,
): PluginErrorDetail | null {
  if (!error || typeof error !== "object") return null;

  // Axios error: error.response.data.detail
  const axiosError = error as {
    response?: { data?: { detail?: unknown } };
    detail?: unknown;
  };

  const detail = axiosError.response?.data?.detail ?? axiosError.detail;
  if (isPluginErrorDetail(detail)) return detail;

  // Some HTTP clients flatten the response body
  const directDetail = error as { detail?: unknown };
  if (isPluginErrorDetail(directDetail.detail)) return directDetail.detail;

  return null;
}
