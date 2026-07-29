import { ConversationClient } from "@openhands/typescript-client/clients";
import { RemoteEventsList } from "@openhands/typescript-client/events/remote-events-list";
import { OpenHandsEvent } from "#/types/agent-server/core";
import { buildHttpBaseUrl } from "#/utils/websocket-url";
import { getActiveBackend } from "../backend-registry/active-store";
import { callCloudProxy } from "../cloud/proxy";
import {
  getAgentServerClientOptions,
  getAgentServerHttpClientOptions,
} from "../agent-server-client-options";
import type {
  ConfirmationResponseRequest,
  ConfirmationResponseResponse,
  EventSearchOptions,
  EventSearchPage,
} from "./event-service.types";

/**
 * Cloud-mode REST calls are split between two upstream hosts (matching
 * OpenHands' cloud frontend):
 *
 *   - **App API** (`backend.host`, default in `callCloudProxy`):
 *     event *history* (`/api/v1/conversation/{id}/events/search`).
 *     Persisted by the cloud backend — survives the runtime sandbox.
 *
 *   - **Runtime sandbox** (extracted from `conversation.conversation_url`
 *     and passed as `hostOverride`): live runtime endpoints like
 *     `/api/conversations/{id}/events/count` and
 *     `/api/conversations/{id}/events/respond_to_confirmation`. Auth on
 *     these endpoints is `X-Session-API-Key`, not `Authorization: Bearer`.
 *
 * App API calls go directly to the cloud backend with bearer auth. Runtime
 * sandbox calls go through `/api/cloud-proxy`, which avoids depending on CORS
 * for per-conversation runtime hosts.
 *
 * Local mode keeps the existing typescript-client path: it targets the
 * conversation's host directly via typed client classes.
 */
class EventService {
  static async respondToConfirmation(
    conversationId: string,
    conversationUrl: string,
    request: ConfirmationResponseRequest,
    sessionApiKey?: string | null,
  ): Promise<ConfirmationResponseResponse> {
    const active = getActiveBackend().backend;

    if (active.kind === "cloud") {
      return callCloudProxy<ConfirmationResponseResponse>({
        backend: active,
        method: "POST",
        hostOverride: buildHttpBaseUrl(conversationUrl),
        path: `/api/conversations/${conversationId}/events/respond_to_confirmation`,
        body: request,
        authMode: "session-api-key",
        sessionApiKey,
      });
    }

    return new ConversationClient(
      getAgentServerClientOptions({
        conversationUrl,
        sessionApiKey,
      }),
    ).respondToConfirmation<ConfirmationResponseResponse>(
      conversationId,
      request,
    );
  }

  static async getEventCount(
    conversationId: string,
    conversationUrl: string,
    sessionApiKey?: string | null,
  ): Promise<number> {
    const active = getActiveBackend().backend;

    if (active.kind === "cloud") {
      return callCloudProxy<number>({
        backend: active,
        method: "GET",
        hostOverride: buildHttpBaseUrl(conversationUrl),
        path: `/api/conversations/${conversationId}/events/count`,
        authMode: "session-api-key",
        sessionApiKey,
      });
    }

    return new ConversationClient(
      getAgentServerClientOptions({
        conversationUrl,
        sessionApiKey,
      }),
    ).getEventCount(conversationId);
  }

  /**
   * Search events for a conversation. Returns the raw page so callers can
   * paginate (via `next_page_id`) and so REST-driven history loading can
   * tell when there are no more older events to load.
   */
  static async searchEvents(
    conversationId: string,
    conversationUrl?: string | null,
    sessionApiKey?: string | null,
    options: EventSearchOptions = {},
  ): Promise<EventSearchPage<OpenHandsEvent>> {
    const active = getActiveBackend().backend;
    const limit = options.limit ?? 100;

    if (active.kind === "cloud") {
      // Event *history* lives on the cloud App API, not the runtime
      // sandbox. Path is singular `conversation` and v1-prefixed.
      //
      // Full pagination params (sort_order, page_id, timestamp filters)
      // require the server-side fix from OpenHands/OpenHands#14399. If
      // the cloud backend hasn't been updated yet, the timestamp filters
      // trigger a 500 (str-vs-datetime comparison). We attempt the full
      // request first and fall back to a limit-only request on failure.
      const cloudLimit = Math.min(limit, 100);
      const hasFilterParams = !!(
        options.sortOrder ||
        options.pageId ||
        options.timestampGte ||
        options.timestampLt
      );

      const params = new URLSearchParams();
      params.set("limit", String(cloudLimit));
      if (options.sortOrder) params.set("sort_order", options.sortOrder);
      if (options.pageId) params.set("page_id", options.pageId);
      if (options.timestampGte)
        params.set("timestamp__gte", options.timestampGte);
      if (options.timestampLt) params.set("timestamp__lt", options.timestampLt);

      const doCloudSearch = (searchParams: URLSearchParams) =>
        callCloudProxy<EventSearchPage<OpenHandsEvent>>({
          backend: active,
          method: "GET",
          path: `/api/v1/conversation/${conversationId}/events/search?${searchParams.toString()}`,
        });

      try {
        const data = await doCloudSearch(params);
        return {
          items: data?.items ?? [],
          next_page_id: data?.next_page_id ?? null,
        };
      } catch (err) {
        if (!hasFilterParams) throw err;
        if (options.strictPagination) throw err;

        // Server doesn't support timestamp filters yet — stop pagination
        // by returning an empty page so the UI doesn't retry indefinitely.
        // A limit-only fallback would return the same most-recent events
        // already in the store, which get deduped but keep hasMore=true.
        console.warn(
          "[EventService] Cloud backend doesn't support pagination filters. " +
            "Falling back to initial load only. " +
            "Server needs OpenHands/OpenHands#14399.",
        );
        return { items: [], next_page_id: null };
      }
    }

    const page = await new RemoteEventsList(
      getAgentServerHttpClientOptions({ conversationUrl, sessionApiKey }),
      conversationId,
    ).search({
      limit,
      ...(options.pageId ? { page_id: options.pageId } : {}),
      ...(options.sortOrder ? { sort_order: options.sortOrder } : {}),
      ...(options.timestampGte ? { timestamp__gte: options.timestampGte } : {}),
      ...(options.timestampLt ? { timestamp__lt: options.timestampLt } : {}),
    });

    return {
      items: (page?.items ?? []) as OpenHandsEvent[],
      next_page_id: page?.next_page_id ?? null,
    };
  }
}

export default EventService;
