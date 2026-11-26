import type { OpenHandsEvent } from "#/types/agent-server/core";
import {
  isExecuteBashActionEvent,
  isExecuteBashObservationEvent,
} from "#/types/agent-server/type-guards";

export type Command = { content: string; type: "input" | "output" };

export function getTerminalCommands(events: OpenHandsEvent[]): Command[] {
  return events.flatMap((event): Command[] => {
    if (isExecuteBashActionEvent(event)) {
      return [{ content: event.action.command, type: "input" }];
    }
    if (isExecuteBashObservationEvent(event)) {
      return [
        {
          content: event.observation.content
            .filter((content) => content.type === "text")
            .map((content) => content.text)
            .join("\n"),
          type: "output",
        },
      ];
    }
    return [];
  });
}
