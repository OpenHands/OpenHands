import { useTranslation } from "react-i18next";
import { useTerminal } from "#/hooks/use-terminal";
import "@xterm/xterm/css/xterm.css";
import { RUNTIME_INACTIVE_STATES } from "#/types/agent-state";
import { cn } from "#/utils/utils";
import { WaitingForRuntimeMessage } from "../chat/waiting-for-runtime-message";
import { useAgentState } from "#/hooks/use-agent-state";
import { I18nKey } from "#/i18n/declaration";

function Terminal() {
  const { t } = useTranslation();
  const { curAgentState, isArchived } = useAgentState();

  // Don't show runtime inactive state for archived conversations
  const isRuntimeInactive =
    !isArchived && RUNTIME_INACTIVE_STATES.includes(curAgentState);

  const ref = useTerminal();

  return (
    <div className="h-full flex flex-col rounded-xl">
      {isArchived && (
        <div className="w-full h-full flex items-center text-center justify-center text-2xl text-tertiary-light pt-16">
          {t(I18nKey.CONVERSATION$ARCHIVED_READ_ONLY)}
        </div>
      )}
      {!isArchived && isRuntimeInactive && (
        <WaitingForRuntimeMessage className="pt-16" />
      )}

      <div className="flex-1 min-h-0 p-4">
        <div
          ref={ref}
          className={cn(
            "w-full h-full",
            isRuntimeInactive || isArchived
              ? "p-0 w-0 h-0 opacity-0 overflow-hidden"
              : "",
          )}
        />
      </div>
    </div>
  );
}

export default Terminal;
