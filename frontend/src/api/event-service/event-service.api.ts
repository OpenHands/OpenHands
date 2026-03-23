import axios from "axios";
import { buildHttpBaseUrl } from "#/utils/websocket-url";
import { buildSessionHeaders } from "#/utils/utils";
import type {
  ConfirmationResponseRequest,
  ConfirmationResponseResponse,
} from "./event-service.types";
import { openHands } from "../open-hands-axios";
import { OpenHandsEvent } from "#/types/v1/core";
import type { V1SendMessageRequest } from "#/api/conversation-service/v1-conversation-service.types";

class EventService {
  /**
   * Respond to a confirmation request in a V1 conversation
   * @param conversationId The conversation ID
   * @param conversationUrl The conversation URL (e.g., "http://localhost:54928/api/conversations/...")
   * @param request The confirmation response request
   * @param sessionApiKey Session API key for authentication (required for V1)
   * @returns The confirmation response
   */
  static async respondToConfirmation(
    conversationId: string,
    conversationUrl: string,
    request: ConfirmationResponseRequest,
    sessionApiKey?: string | null,
  ): Promise<ConfirmationResponseResponse> {
    // Build the runtime URL using the conversation URL
    const runtimeUrl = buildHttpBaseUrl(conversationUrl);

    // Build session headers for authentication
    const headers = buildSessionHeaders(sessionApiKey);

    // Make the API call to the runtime endpoint
    const { data } = await axios.post<ConfirmationResponseResponse>(
      `${runtimeUrl}/api/conversations/${conversationId}/events/respond_to_confirmation`,
      request,
      { headers },
    );

    return data;
  }

  /**
   * Get event count for a V1 conversation
   * @param conversationId The conversation ID
   * @param conversationUrl The conversation URL (e.g., "http://localhost:54928/api/conversations/...")
   * @param sessionApiKey Session API key for authentication (required for V1)
   * @returns The event count
   */
  static async getEventCount(
    conversationId: string,
    conversationUrl: string,
    sessionApiKey?: string | null,
  ): Promise<number> {
    // Build the runtime URL using the conversation URL
    const runtimeUrl = buildHttpBaseUrl(conversationUrl);

    // Build session headers for authentication
    const headers = buildSessionHeaders(sessionApiKey);

    const { data } = await axios.get<number>(
      `${runtimeUrl}/api/conversations/${conversationId}/events/count`,
      { headers },
    );
    return data;
  }

  /**
   * Post a message event directly to the agent server.
   * Used when the WebSocket is closed but the conversation is already running.
   *
   * @param conversationId The agent server conversation ID
   * @param conversationUrl The conversation URL containing the agent server base
   * @param message The message to send
   * @param sessionApiKey Session API key for authentication
   */
  static async postEvent(
    conversationId: string,
    conversationUrl: string,
    message: V1SendMessageRequest,
    sessionApiKey?: string | null,
  ): Promise<void> {
    const runtimeUrl = buildHttpBaseUrl(conversationUrl);
    const headers = buildSessionHeaders(sessionApiKey);
    await axios.post(
      `${runtimeUrl}/api/conversations/${conversationId}/events`,
      { role: message.role, content: message.content, run: true },
      { headers },
    );
  }

  // V1 conversations — App Server REST endpoint
  static async searchEventsV1(conversationId: string, limit = 100) {
    const { data } = await openHands.get<{
      items: OpenHandsEvent[];
    }>(`/api/v1/conversation/${conversationId}/events/search`, {
      params: { limit },
    });

    return data.items;
  }

  // V0 conversations — Legacy REST endpoint
  static async searchEventsV0(conversationId: string, limit = 100) {
    const { data } = await openHands.get<{
      events: OpenHandsEvent[];
    }>(`/api/conversations/${conversationId}/events`, {
      params: { limit },
    });

    return data.events;
  }
}
export default EventService;
