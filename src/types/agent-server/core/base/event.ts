import {
  EventID,
  SourceType,
  ToolCallID,
  TextContent,
  ImageContent,
} from "./common";

// Base event interface - fundamental properties for all events
export interface BaseEvent {
  /**
   * Unique event id (ULID/UUID)
   */
  id: EventID;

  /**
   * Event timestamp (ISO string)
   */
  timestamp: string;

  /**
   * The source of this event
   */
  source: SourceType;
}

// LLM Message structure
export interface Message {
  role: "user" | "system" | "assistant" | "tool";
  content: (TextContent | ImageContent)[];
  cache_enabled?: boolean;
  vision_enabled?: boolean;
  tool_calls?: ChatCompletionMessageToolCall[];
  reasoning_content?: string | null;
  thinking_blocks?: (ThinkingBlock | RedactedThinkingBlock)[];
  name?: string;
  tool_call_id?: ToolCallID;
}

// Tool call structure from LiteLLM
export interface ChatCompletionMessageToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

// Tool parameter structure from LiteLLM
export interface ChatCompletionToolParam {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

// Thinking blocks for Anthropic extended thinking feature
export interface ThinkingBlock {
  type: "thinking";
  /**
   * The thinking content
   */
  thinking: string;
  /**
   * Cryptographic signature for the thinking block
   */
  signature: string;
}

export interface RedactedThinkingBlock {
  type: "redacted_thinking";
  /**
   * The redacted thinking content
   */
  data: string;
}
