import { useTerminal } from "#/hooks/use-terminal";
import "@xterm/xterm/css/xterm.css";
import { RUNTIME_INACTIVE_STATES } from "#/types/agent-state";
import { cn } from "#/utils/utils";
import { WaitingForRuntimeMessage } from "../chat/waiting-for-runtime-message";
import { useAgentState } from "#/hooks/use-agent-state";

function Terminal() {
  const { curAgentState } = useAgentState();

  const isRuntimeInactive = RUNTIME_INACTIVE_STATES.includes(curAgentState);

  const ref = useTerminal();

  return (
    <div className="h-full flex flex-col rounded-xl">
      {isRuntimeInactive && <WaitingForRuntimeMessage className="pt-16" />}

      <div className="flex-1 min-h-0 p-4">
        <div
          ref={ref}
          className={cn(
            "w-full h-full",
            isRuntimeInactive ? "p-0 w-0 h-0 opacity-0 overflow-hidden" : "",
          )}
        />
      </div>
    </div>
  );
}

export default Terminal;
