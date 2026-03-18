/**
 * Types for the pending message service
 */

import type { V1MessageContent } from "../conversation-service/v1-conversation-service.types";

/**
 * Response when queueing a pending message
 */
export interface PendingMessageResponse {
  id: string;
  queued: boolean;
  position: number;
}

/**
 * Request to queue a pending message
 */
export interface QueuePendingMessageRequest {
  role?: "user";
  content: V1MessageContent[];
}
