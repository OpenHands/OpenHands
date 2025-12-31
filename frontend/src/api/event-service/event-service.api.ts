import axios from "axios";
import { buildHttpBaseUrl } from "#/utils/websocket-url";
import { buildSessionHeaders } from "#/utils/utils";
import type {
  ConfirmationResponseRequest,
  ConfirmationResponseResponse,
} from "./event-service.types";

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
}
export default EventService;
