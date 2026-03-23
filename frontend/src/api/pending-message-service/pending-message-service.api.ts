/**
 * Pending Message Service
 *
 * This service handles server-side message queuing for V1 conversations.
 * Messages can be queued when the WebSocket is not connected and will be
 * delivered automatically when the conversation becomes ready.
 */

import { openHands } from "../open-hands-axios";
import type {
  PendingMessageResponse,
  QueuePendingMessageRequest,
} from "./pending-message-service.types";

class PendingMessageService {
  /**
   * Queue a message for delivery when conversation becomes ready.
   *
   * This endpoint allows users to submit messages even when the conversation's
   * WebSocket connection is not yet established. Messages are stored server-side
   * and delivered automatically when the conversation transitions to READY status.
   *
   * @param conversationId The conversation ID (can be task ID before conversation is ready)
   * @param message The message to queue
   * @returns PendingMessageResponse with the message ID and queue position
   * @throws Error if too many pending messages (limit: 10 per conversation)
   */
  static async queueMessage(
    conversationId: string,
    message: QueuePendingMessageRequest,
  ): Promise<PendingMessageResponse> {
    const { data } = await openHands.post<PendingMessageResponse>(
      `/api/v1/conversations/${conversationId}/pending-messages`,
      message,
    );
    return data;
  }
}

export default PendingMessageService;
