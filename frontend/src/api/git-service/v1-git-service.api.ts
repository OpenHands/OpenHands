import axios from "axios";
import { buildHttpBaseUrl } from "#/utils/websocket-url";
import { buildSessionHeaders } from "#/utils/utils";
import { mapV1ToV0Status } from "#/utils/git-status-mapper";
import type {
  GitChange,
  GitChangeDiff,
  V1GitChangeStatus,
} from "../open-hands.types";

interface V1GitChange {
  status: V1GitChangeStatus;
  path: string;
}

class V1GitService {
  /**
   * Build the full URL for V1 runtime-specific endpoints
   * @param conversationUrl The conversation URL (e.g., "http://localhost:54928/api/conversations/...")
   * @param path The API path (e.g., "/api/git/changes")
   * @returns Full URL to the runtime endpoint
   */
  private static buildRuntimeUrl(
    conversationUrl: string | null | undefined,
    path: string,
  ): string {
    const baseUrl = buildHttpBaseUrl(conversationUrl);
    return `${baseUrl}${path}`;
  }

  /**
   * Get git changes for a V1 conversation
   * Uses the agent server endpoint: GET /api/git/changes/{path}
   * Maps V1 status types (ADDED, DELETED, etc.) to V0 format (A, D, etc.)
   *
   * @param conversationUrl The conversation URL (e.g., "http://localhost:54928/api/conversations/...")
   * @param sessionApiKey Session API key for authentication (required for V1)
   * @param path The git repository path (e.g., /workspace/project or /workspace/project/OpenHands)
   * @returns List of git changes with V0-compatible status types
   */
  static async getGitChanges(
    conversationUrl: string | null | undefined,
    sessionApiKey: string | null | undefined,
    path: string,
  ): Promise<GitChange[]> {
    const encodedPath = encodeURIComponent(path);
    const url = this.buildRuntimeUrl(
      conversationUrl,
      `/api/git/changes/${encodedPath}`,
    );
    const headers = buildSessionHeaders(sessionApiKey);

    // V1 API returns V1GitChangeStatus types, we need to map them to V0 format
    const { data } = await axios.get<V1GitChange[]>(url, { headers });

    // Validate response is an array (could be HTML error page if runtime is dead)
    if (!Array.isArray(data)) {
      throw new Error(
        "Invalid response from runtime - runtime may be unavailable",
      );
    }

    // Map V1 statuses to V0 format for compatibility
    return data.map((change) => ({
      status: mapV1ToV0Status(change.status),
      path: change.path,
    }));
  }

  /**
   * Get git change diff for a specific file in a V1 conversation
   * Uses the agent server endpoint: GET /api/git/diff/{path}
   *
   * @param conversationUrl The conversation URL (e.g., "http://localhost:54928/api/conversations/...")
   * @param sessionApiKey Session API key for authentication (required for V1)
   * @param path The file path to get diff for
   * @returns Git change diff
   */
  static async getGitChangeDiff(
    conversationUrl: string | null | undefined,
    sessionApiKey: string | null | undefined,
    path: string,
  ): Promise<GitChangeDiff> {
    const encodedPath = encodeURIComponent(path);
    const url = this.buildRuntimeUrl(
      conversationUrl,
      `/api/git/diff/${encodedPath}`,
    );
    const headers = buildSessionHeaders(sessionApiKey);

    const { data } = await axios.get<GitChangeDiff>(url, { headers });
    return data;
  }
}

export default V1GitService;
