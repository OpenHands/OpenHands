import { TextContent } from "../base/common";
import { CriticResult } from "../base/critic";
import { BaseEvent, Message } from "../base/event";

export interface MessageEvent extends BaseEvent {
  /**
   * The exact LLM message for this message event
   */
  llm_message: Message;

  /**
   * List of activated microagent names
   */
  activated_microagents: string[];

  /**
   * List of content added by agent context
   */
  extended_content: TextContent[];

  /**
   * Optional critic evaluation of the agent's work at this point.
   */
  critic_result?: CriticResult | null;
}
