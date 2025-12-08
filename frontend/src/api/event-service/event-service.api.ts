import axios from "axios";
import { buildHttpBaseUrl } from "#/utils/websocket-url";
import { buildSessionHeaders } from "#/utils/utils";
import type {
  ConfirmationResponseRequest,
  ConfirmationResponseResponse,
} from "./event-service.types";
import { openHands } from "../open-hands-axios";

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

  static async getEventCount(conversationId: string): Promise<number> {
    const params = new URLSearchParams();
    params.append("conversation_id__eq", conversationId);
    const { data } = await openHands.get<number>(
      `/api/v1/events/count?${params.toString()}`,
    );
    return data;
  }
}
export default EventService;
