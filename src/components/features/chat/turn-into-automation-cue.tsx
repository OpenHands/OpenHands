import { useTranslation } from "react-i18next";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useAutomationHealth } from "#/hooks/query/use-automation-health";
import { useTracking } from "#/hooks/use-tracking";
import { I18nKey } from "#/i18n/declaration";
import AutomationsIcon from "#/icons/automations.svg?react";
import { useConversationStore } from "#/stores/conversation-store";

interface TurnIntoAutomationCueProps {
  /** The agent has handed back a turn that contains work worth repeating. */
  isTurnComplete: boolean;
}

/**
 * End-of-turn entry point for turning THIS conversation into an automation.
 *
 * Clicking only drafts a request in this conversation's composer, so the agent
 * that answers it already holds the whole workflow as context. The user can
 * edit the request before sending, and it asks the agent to show its draft for
 * confirmation — nothing is sent, created or enabled by the click itself. A
 * message the user is already typing is left alone.
 */
export function TurnIntoAutomationCue({
  isTurnComplete,
}: TurnIntoAutomationCueProps) {
  const { t } = useTranslation("openhands");
  const active = useActiveBackend();
  const { data: health } = useAutomationHealth();
  const { trackAutomationCreatedButton } = useTracking();
  const restoreMessageToInputIfEmpty = useConversationStore(
    (state) => state.restoreMessageToInputIfEmpty,
  );

  if (!isTurnComplete || health?.status !== "ok") return null;

  const handleClick = () => {
    trackAutomationCreatedButton({
      backendKind: active.backend.kind,
      source: "conversation",
    });
    restoreMessageToInputIfEmpty(
      t(I18nKey.AUTOMATIONS$TURN_INTO_AUTOMATION_PROMPT),
    );
  };

  return (
    <div className="flex px-1">
      <button
        type="button"
        data-testid="turn-into-automation-cue"
        onClick={handleClick}
        className="flex min-w-0 max-w-full items-center gap-1.5 rounded-full border border-border bg-surface px-3 py-1.5 text-xs text-text-secondary transition-colors hover:bg-surface-raised hover:text-foreground"
      >
        <AutomationsIcon
          width={14}
          height={14}
          className="shrink-0"
          aria-hidden
        />
        <span className="min-w-0 truncate">
          {t(I18nKey.AUTOMATIONS$TURN_INTO_AUTOMATION)}
        </span>
      </button>
    </div>
  );
}
