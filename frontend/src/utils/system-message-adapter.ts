import { OHEvent } from "#/stores/use-event-store";
import { isActionOrObservation, isSystemMessage } from "#/types/core/guards";
import { ChatCompletionToolParam } from "#/types/v1/core";
import {
  isSystemPromptEvent,
  isV0Event,
  isV1Event,
} from "#/types/v1/type-guards";

export interface SystemMessageForModal {
  content: string;
  tools: ChatCompletionToolParam[] | Record<string, unknown>[] | null;
  openhands_version: string | null;
  agent_class: string | null;
}

export function adaptSystemMessage(
  events: OHEvent[],
): SystemMessageForModal | null {
  let systemMessage: SystemMessageForModal | null = null;
  // 여기의 events타입은 2가지가 들어올 수 있다. v0와 v1타입
  const v0SystemMessage = events
    .filter(isV0Event)
    .filter(isActionOrObservation)
    .find(isSystemMessage);

  // V1 System Prompt Event
  const v1SystemPromptEvent = events
    .filter(isV1Event)
    .find(isSystemPromptEvent);

  if (v0SystemMessage) {
    systemMessage = v0SystemMessage.args;
  } else if (v1SystemPromptEvent) {
    systemMessage = {
      content: v1SystemPromptEvent.system_prompt.text,
      tools: v1SystemPromptEvent.tools ?? null,
      openhands_version: null,
      agent_class: null,
    };
  }

  if (systemMessage) {
    return systemMessage;
  }

  return null;
}
