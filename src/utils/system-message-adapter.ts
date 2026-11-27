import { OHEvent } from "#/stores/use-event-store";
import { ChatCompletionToolParam } from "#/types/agent-server/core";
import { isSystemPromptEvent } from "#/types/agent-server/type-guards";
import { redactCustomSecrets } from "#/utils/redact-custom-secrets";

export interface SystemMessageForModal {
  content: string;
  tools: ChatCompletionToolParam[] | Record<string, unknown>[] | null;
  openhands_version: string | null;
  agent_class: string | null;
}

export function adaptSystemMessage(
  events: OHEvent[],
): SystemMessageForModal | null {
  const systemPromptEvent = events.find(isSystemPromptEvent);

  if (!systemPromptEvent) {
    return null;
  }

  // dynamic_context is the runtime-injected tail of the same system prompt the
  // model receives, so append it to show the full message as one block.
  const dynamicContextText = systemPromptEvent.dynamic_context?.text;
  const content = dynamicContextText
    ? `${systemPromptEvent.system_prompt.text.trimEnd()}\n\n${redactCustomSecrets(dynamicContextText)}`
    : systemPromptEvent.system_prompt.text;

  return {
    content,
    tools: systemPromptEvent.tools ?? null,
    openhands_version: null,
    agent_class: null,
  };
}
