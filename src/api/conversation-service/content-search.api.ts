import axios from "axios";
import { getTelemetryDistinctId } from "#/services/telemetry";
import {
  AGENT_CANVAS_CLIENT_HEADERS,
  OPENHANDS_TELEMETRY_DISTINCT_ID_HEADER,
} from "../client-source";
import { getEffectiveLocalBackend } from "../backend-registry/active-store";
import { NoBackendAvailableError } from "../agent-server-client-options";

// Content search is a local agent-server feature: the endpoint lives on the
// local backend only, so cloud users must never see it. The guard test
// (no-direct-agent-server-calls.test.ts) forbids ad-hoc HTTP except for
// allow-listed files; this file follows the automation-service precedent:
// a dedicated axios instance that resolves the *local* backend from the
// registry on every call (so it tracks host/key edits made in the
// manage-backends UI) and attaches the session key.

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
  title: string;
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
