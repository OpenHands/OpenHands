/**
 * @deprecated This type is deprecated and will be removed in a future version.
 * Use agent-server event types instead.
 */
export interface ActionMessage {
  id: number;

  // Either 'agent' or 'user'
  source: "agent" | "user";

  // The action to be taken
  action: string;

  // The arguments for the action
  args: Record<string, string>;

  // A friendly message that can be put in the chat log
  message: string;

  // The timestamp of the message
  timestamp: string;

  // LLM metrics information
  llm_metrics?: {
    accumulated_cost: number;
    max_budget_per_task: number | null;
    accumulated_token_usage: {
      prompt_tokens: number;
      completion_tokens: number;
      cache_read_tokens: number;
      cache_write_tokens: number;
      context_window: number;
      per_turn_token: number;
    };
  };

  // Tool call metadata
  tool_call_metadata?: {
    model_response?: {
      usage: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
      };
    };
  };
}

export interface ObservationMessage {
  // The type of observation
  observation: string;

  id: number;
  cause: number;

  // The observed data
  content: string;

  extras: {
    metadata: Record<string, unknown>;
    error_id: string;
    [key: string]: string | Record<string, unknown>;
  };

  // A friendly message that can be put in the chat log
  message: string;

  // The timestamp of the message
  timestamp: string;
}

export interface StatusMessage {
  status_update: true;
  type: string;
  id?: string;
  message: string;
  conversation_title?: string;
}
