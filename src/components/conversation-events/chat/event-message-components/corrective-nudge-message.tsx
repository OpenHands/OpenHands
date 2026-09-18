import React from 'react';
import { MessageEvent } from '#/types/agent-server/core';
import InfoCircleIcon from '#/icons/info-circle.svg?react';
import { CORRECTIVE_NUDGE_TEXT } from '../event-content-helpers/should-render-event';
import { parseMessageFromEvent } from '../event-content-helpers/parse-message-from-event';

interface CorrectiveNudgeMessageProps {
  event: MessageEvent;
}

/**
 * Renders the SDK's empty-response corrective nudge as a distinct
 * informational message — italic, muted, with an info icon —
 * so it is clearly a system/framework note rather than user input.
 */
export function CorrectiveNudgeMessage({ event }: CorrectiveNudgeMessageProps) {
  const text = parseMessageFromEvent(event);

  return (
    <div
      data-testid="corrective-nudge"
      className="mt-6 flex w-fit max-w-full items-start gap-2 self-start rounded-xl bg-tertiary/40 px-4 py-2.5 italic text-muted-foreground last:mb-4"
    >
      <InfoCircleIcon
        className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground"
        aria-hidden
      />
      <span className="text-sm leading-6 whitespace-normal [word-break:break-word]">
        {text || CORRECTIVE_NUDGE_TEXT}
      </span>
    </div>
  );
}
