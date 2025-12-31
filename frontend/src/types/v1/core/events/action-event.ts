import { Action } from "../base/action";
import { EventID, ToolCallID, SecurityRisk, TextContent } from "../base/common";
import {
  BaseEvent,
  ChatCompletionMessageToolCall,
  ThinkingBlock,
  RedactedThinkingBlock,
} from "../base/event";

export interface ActionEvent<T extends Action = Action> extends BaseEvent {
  /**
   * The thought process of the agent before taking this action
   */
  thought: TextContent[];

  /**
   * Intermediate reasoning/thinking content from reasoning models
   */
  reasoning_content?: string | null;

  /**
   * Anthropic thinking blocks from the LLM response
   */
  thinking_blocks: (ThinkingBlock | RedactedThinkingBlock)[];

  /**
   * Single action (tool call) returned by LLM
   */
  action: T;

  /**
   * The name of the tool being called
   */
  tool_name: string;

  /**
   * The unique id returned by LLM API for this tool call
   */
  tool_call_id: ToolCallID;

  /**
   * The tool call received from the LLM response. We keep a copy of it
   * so it is easier to construct it into LLM message.
   * This could be different from `action`: e.g., `tool_call` may contain
   * `security_risk` field predicted by LLM when LLM risk analyzer is enabled,
   * while `action` does not.
   */
  tool_call: ChatCompletionMessageToolCall;

  /**
   * Groups related actions from same LLM response. This helps in tracking
   * and managing results of parallel function calling from the same LLM
   * response.
   */
  llm_response_id: EventID;

  /**
   * The LLM's assessment of the safety risk of this action
   */
  security_risk: SecurityRisk;
}
