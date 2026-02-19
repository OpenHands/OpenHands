import { BaseEvent } from "../base/event";
import { V1ExecutionStatus } from "../base/common";

/**
 * Token usage metrics for LLM calls
 */
export interface TokenUsage {
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  reasoning_tokens: number;
  context_window: number;
  per_turn_token: number;
  response_id: string;
}

/**
 * LLM metrics for a specific component (agent or condenser)
 */
export interface LLMMetrics {
  model_name: string;
  accumulated_cost: number;
  max_budget_per_task: number | null;
  accumulated_token_usage: TokenUsage;
  costs: Array<{
    model: string;
    cost: number;
    timestamp: number;
  }>;
  response_latencies: Array<{
    model: string;
    latency: number;
    response_id: string;
  }>;
  token_usages: TokenUsage[];
}

/**
 * Usage metrics mapping for different components
 */
export interface UsageToMetrics {
  agent: LLMMetrics;
  condenser: LLMMetrics;
}

/**
 * Stats containing usage metrics
 */
export interface ConversationStats {
  usage_to_metrics: UsageToMetrics;
}

/**
 * Conversation state value types
 */
export interface ConversationState {
  execution_status: V1ExecutionStatus;
  stats?: ConversationStats;
  // Add other conversation state fields here as needed
}

interface ConversationStateUpdateEventBase extends BaseEvent {
  /**
   * Discriminator field for type guards
   */
  kind: "ConversationStateUpdateEvent";

  /**
   * The source is always "environment" for conversation state update events
   */
  source: "environment";

  /**
   * Unique key for this state update event.
   * Can be "full_state" for full state snapshots or field names for partial updates.
   */
  key: "full_state" | "execution_status" | "stats"; // Extend with other keys as needed

  /**
   * Conversation state updates
   */
  value: ConversationState | V1ExecutionStatus | ConversationStats;
}

// Narrowed interfaces for full state update event
export interface ConversationStateUpdateEventFullState extends ConversationStateUpdateEventBase {
  key: "full_state";
  value: ConversationState;
}

// Narrowed interface for agent status update event
export interface ConversationStateUpdateEventAgentStatus extends ConversationStateUpdateEventBase {
  key: "execution_status";
  value: V1ExecutionStatus;
}

// Narrowed interface for stats update event
export interface ConversationStateUpdateEventStats extends ConversationStateUpdateEventBase {
  key: "stats";
  value: ConversationStats;
}

// Conversation state update event - contains conversation state updates
export type ConversationStateUpdateEvent =
  | ConversationStateUpdateEventFullState
  | ConversationStateUpdateEventAgentStatus
  | ConversationStateUpdateEventStats;

// Conversation error event - contains error information
export interface ConversationErrorEvent extends BaseEvent {
  /**
   * Discriminator field for type guards
   */
  kind: "ConversationErrorEvent";

  /**
   * The source is always "environment" for conversation error events
   */
  source: "environment";

  /**
   * Error code (e.g., "AuthenticationError")
   */
  code: string;

  /**
   * Detailed error message
   */
  detail: string;
}
