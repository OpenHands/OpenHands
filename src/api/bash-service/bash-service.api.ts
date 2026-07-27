import { BashClient } from "@openhands/typescript-client/clients";
import type {
  BashEvent,
  BashEventPage,
  BashOutput,
} from "@openhands/typescript-client";
import { buildHttpBaseUrl } from "#/utils/websocket-url";
import { getActiveBackend } from "../backend-registry/active-store";
import { callCloudProxy } from "../cloud/proxy";
import { getAgentServerClientOptions } from "../agent-server-client-options";

interface SearchOptions {
  kind__eq?: "BashCommand" | "BashOutput";
  command_id__eq?: string;
  sort_order?: "TIMESTAMP" | "TIMESTAMP_DESC";
  page_id?: string;
  limit?: number;
}

const MAX_OUTPUT_PAGES = 20; // safety cap; >2000 output events is unlikely.

function isBashOutput(event: BashEvent): event is BashOutput {
  return event.kind === "BashOutput";
}

/**
 * Cloud-aware bash event reads.
 *
 * Bash events live on the agent-server runtime that owns the
 * conversation. In **local** mode we talk to the active backend's
 * agent-server directly with the SDK's `BashClient` (a per-conversation
 * URL is honoured when known, otherwise we fall back to the backend
 * host — a single local agent-server hosts all conversations). In
 * **cloud** mode we tunnel through `callCloudProxy` with the runtime URL
 * as `hostOverride`: direct browser calls to `*.prod-runtime.all-hands.dev`
 * are blocked by CORS, and runtime endpoints authenticate with the
 * conversation's `X-Session-API-Key`.
 *
 * Note on the search filter name: the agent-server API uses
 * `command_id__eq` (not `bash_command_id__eq`) — that's the parameter the
 * `BashService.search_bash_events` Python implementation declares and
 * what the typescript-client's `BashEventSearchOptions` exposes.
 */
class BashService {
  /**
   * Fetch all `BashOutput` events for a bash command, paginated and
   * sorted by timestamp. Returns events in command-emission order so
   * callers can concatenate `stdout` / `stderr` values directly.
   */
  static async listOutputs(
    conversationUrl: string | null,
    sessionApiKey: string | null | undefined,
    bashCommandId: string,
  ): Promise<BashOutput[]> {
    const outputs: BashOutput[] = [];
    let pageId: string | undefined;
    for (let i = 0; i < MAX_OUTPUT_PAGES; i += 1) {
      const page = await BashService.searchEvents(
        conversationUrl,
        sessionApiKey,
        {
          kind__eq: "BashOutput",
          command_id__eq: bashCommandId,
          sort_order: "TIMESTAMP",
          ...(pageId ? { page_id: pageId } : {}),
        },
      );
      page.items.forEach((event) => {
        if (isBashOutput(event)) outputs.push(event);
      });
      if (!page.next_page_id) break;
      pageId = page.next_page_id;
    }
    return outputs;
  }

  private static async searchEvents(
    conversationUrl: string | null,
    sessionApiKey: string | null | undefined,
    options: SearchOptions,
  ): Promise<BashEventPage> {
    const active = getActiveBackend().backend;

    if (active.kind === "cloud") {
      // Cloud requires the per-conversation runtime URL — there is no
      // shared cloud host that owns bash events. Callers must wait for
      // the conversation to be hydrated before invoking this method on
      // a cloud backend.
      if (!conversationUrl) {
        throw new Error(
          "BashService.listOutputs requires a conversation URL on cloud backends",
        );
      }
      const params = new URLSearchParams();
      Object.entries(options).forEach(([k, v]) => {
        if (v !== undefined && v !== null) params.set(k, String(v));
      });
      return callCloudProxy<BashEventPage>({
        backend: active,
        method: "GET",
        hostOverride: buildHttpBaseUrl(conversationUrl),
        path: `/api/bash/bash_events/search?${params.toString()}`,
        authMode: "session-api-key",
        sessionApiKey,
      });
    }

    // Local mode: the active backend's agent-server hosts the bash
    // events. The optional `conversationUrl` is used when present (lets
    // us target a per-conversation sub-host), otherwise we fall through
    // to `backend.host` via `getAgentServerClientOptions`.
    return new BashClient(
      getAgentServerClientOptions({
        ...(conversationUrl ? { conversationUrl } : {}),
        sessionApiKey,
      }),
    ).searchEvents(options);
  }
}

export default BashService;
