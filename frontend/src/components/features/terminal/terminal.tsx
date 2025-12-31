import React from "react";
import { useTerminal } from "#/hooks/use-terminal";
import "@xterm/xterm/css/xterm.css";
import { RUNTIME_INACTIVE_STATES } from "#/types/agent-state";
import { cn } from "#/utils/utils";
import { WaitingForRuntimeMessage } from "../chat/waiting-for-runtime-message";
import { useAgentState } from "#/hooks/use-agent-state";
import { useElementDimensions } from "#/hooks/use-element-dimensions";

function Terminal() {
  const { curAgentState } = useAgentState();

  const isRuntimeInactive = RUNTIME_INACTIVE_STATES.includes(curAgentState);

  const ref = useTerminal();
  const rootRef = React.useRef<HTMLDivElement>(null);

  // Monitor dimensions in real-time (unused but kept for potential future use)
  useElementDimensions(ref, !isRuntimeInactive);

  return (
    <div
      ref={rootRef}
      className="absolute inset-0 flex flex-col rounded-xl relative"
    >
      {isRuntimeInactive && <WaitingForRuntimeMessage className="pt-16" />}

      <div
        ref={ref}
        className={cn(
          "flex-1 min-h-0 p-4 relative",
          isRuntimeInactive ? "hidden" : "",
        )}
      />
    </div>
  );
}

export default Terminal;
