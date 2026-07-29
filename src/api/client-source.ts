import packageJson from "../../package.json";

export const AGENT_CANVAS_CLIENT_SOURCE = "agent_canvas";
export const AGENT_CANVAS_CLIENT_VERSION = packageJson.version;

/**
 * Coarse, non-user-identifying request metadata for Cloud observability.
 *
 * Cloud ingress can retain these headers as facets without logging request
 * bodies, API keys, device codes, or conversation content.
 */
export const OPENHANDS_CLIENT_HEADER = "X-OpenHands-Client";
export const OPENHANDS_CLIENT_VERSION_HEADER = "X-OpenHands-Client-Version";
export const OPENHANDS_TELEMETRY_DISTINCT_ID_HEADER =
  "X-OpenHands-Telemetry-Distinct-Id";

export const AGENT_CANVAS_CLIENT_HEADERS: Readonly<Record<string, string>> = {
  [OPENHANDS_CLIENT_HEADER]: AGENT_CANVAS_CLIENT_SOURCE,
  [OPENHANDS_CLIENT_VERSION_HEADER]: AGENT_CANVAS_CLIENT_VERSION,
};
