import { MessageEvent } from "#/types/agent-server/core";
import InfoCircleIcon from "#/icons/info-circle.svg?react";
import { parseMessageFromEvent } from "../event-content-helpers/parse-message-from-event";

interface CorrectiveNudgeMessageProps {
  event: MessageEvent;
}

export function CorrectiveNudgeMessage({ event }: CorrectiveNudgeMessageProps) {
  return (
    <div
      role="note"
      data-testid="corrective-nudge-message"
      className="mt-6 flex w-full items-start gap-1.5 text-sm italic text-muted last:mb-4"
    >
      <InfoCircleIcon
        width={14}
        height={14}
        className="mt-[5px] shrink-0"
        aria-hidden
      />
      <span className="leading-6 [word-break:break-word]">
        {parseMessageFromEvent(event)}
      </span>
    </div>
  );
}
