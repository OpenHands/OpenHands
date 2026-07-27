import { BaseEvent } from "../base/event";
import { ExecutionStatus } from "../base/common";

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
  execution_status: ExecutionStatus;
  stats?: ConversationStats;
  // Add other conversation state fields here as needed
}

/**
 * The judge's verdict on whether a `/goal` objective is complete.
 */
export interface GoalVerdict {
  /** Probability (0-1) that the full objective is provably done. */
  score: number;
  /** Whether the judge considers the objective complete. */
  complete: boolean;
  /** Concise description of what remains, or empty if complete. */
  missing: string;
}

/**
 * Live status of a `/goal` loop, streamed as the `value` of a
 * ConversationStateUpdateEvent with `key: "goal"` at each lifecycle point
 * (kickoff, each round, and the terminal/interrupted state).
 */
export interface GoalStatus {
  /** Whether the goal loop is still running. */
  active: boolean;
  status: "running" | "complete" | "capped" | "interrupted";
  /** Audit rounds completed so far (0 at kickoff). */
  iteration: number;
  /**
   * Maximum audit rounds before the loop gives up. snake_case mirrors the
   * agent-server `GoalStatus` payload (this event value is its `model_dump`);
   * renaming it to camelCase would stop matching the streamed event JSON.
   */
  max_iterations: number;
  objective: string;
  /** Last judge verdict; null at kickoff and on an interrupted loop. */
  verdict: GoalVerdict | null;
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
  key: "full_state" | "execution_status" | "stats" | "goal"; // Extend with other keys as needed

  /**
   * Conversation state updates
   */
  value: ConversationState | ExecutionStatus | ConversationStats | GoalStatus;
}

// Narrowed interfaces for full state update event
export interface ConversationStateUpdateEventFullState extends ConversationStateUpdateEventBase {
  key: "full_state";
  value: ConversationState;
}

// Narrowed interface for agent status update event
export interface ConversationStateUpdateEventAgentStatus extends ConversationStateUpdateEventBase {
  key: "execution_status";
  value: ExecutionStatus;
}

// Narrowed interface for stats update event
export interface ConversationStateUpdateEventStats extends ConversationStateUpdateEventBase {
  key: "stats";
  value: ConversationStats;
}

// Narrowed interface for goal status update event
export interface ConversationStateUpdateEventGoal extends ConversationStateUpdateEventBase {
  key: "goal";
  value: GoalStatus;
}

// Conversation state update event - contains conversation state updates
export type ConversationStateUpdateEvent =
  | ConversationStateUpdateEventFullState
  | ConversationStateUpdateEventAgentStatus
  | ConversationStateUpdateEventStats
  | ConversationStateUpdateEventGoal;

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

// Server error event - contains error information
export interface ServerErrorEvent extends BaseEvent {
  /**
   * Discriminator field for type guards
   */
  kind: "ServerErrorEvent";

  /**
   * The source is always "environment" for server error events
   */
  source: "environment";

  /**
   * Error code (e.g., "MCPError")
   */
  code: string;

  /**
   * Detailed error message
   */
  detail: string;
}
