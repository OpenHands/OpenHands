import axios from "axios";
import { getTelemetryDistinctId } from "#/services/telemetry";
import {
  AGENT_CANVAS_CLIENT_HEADERS,
  OPENHANDS_TELEMETRY_DISTINCT_ID_HEADER,
} from "../client-source";
import { getEffectiveLocalBackend } from "../backend-registry/active-store";
import { NoBackendAvailableError } from "../agent-server-client-options";

// Content search is a local agent-server feature; cloud users never see
// it. This dedicated axios instance resolves the *local* backend from the
// registry on every call (so host/key edits made in the manage-backends
// UI apply) and attaches the session key.

const localContentSearchAxios = axios.create();

localContentSearchAxios.interceptors.request.use(async (config) => {
  const telemetryDistinctId = await getTelemetryDistinctId();
  if (telemetryDistinctId) {
    config.headers.set(
      OPENHANDS_TELEMETRY_DISTINCT_ID_HEADER,
      telemetryDistinctId,
    );
  }
  Object.entries(AGENT_CANVAS_CLIENT_HEADERS).forEach(([name, value]) => {
    config.headers.set(name, value);
  });

  const backend = getEffectiveLocalBackend();
  if (!backend) throw new NoBackendAvailableError();

  // eslint-disable-next-line no-param-reassign
  config.baseURL = backend.host;
  const apiKey = backend.apiKey?.trim();
  if (apiKey) {
    config.headers.set("X-Session-API-Key", apiKey);
  }
  return config;
});

export interface ContentSearchHit {
  conversation_id: string;
  title: string | null;
  event_id: string;
  source: string;
  timestamp: string | null;
  snippet: string;
}

export interface ContentSearchResponse {
  results: ContentSearchHit[];
  total: number;
  capped: boolean;
}

export async function searchConversationContent(
  query: string,
  limit: number = 50,
): Promise<ContentSearchResponse> {
  const { data } = await localContentSearchAxios.get<ContentSearchResponse>(
    "/api/conversations/search-content",
    {
      params: { query, limit },
    },
  );
  return data;
}

/**
 * Feature probe: the backend mounts /search/enabled only when the content
 * search service is constructed (OH_CONTENT_SEARCH_ENABLED and local). A
 * 404 or a transport error means the feature is off, so callers treat any
 * failure as "unavailable" — the UI then hides the box instead of showing
 * an error line.
 */
export async function isContentSearchEnabled(): Promise<boolean> {
  try {
    const { data } = await localContentSearchAxios.get<{ enabled: boolean }>(
      "/api/conversations/search/enabled",
    );
    return data?.enabled === true;
  } catch {
    return false;
  }
}
