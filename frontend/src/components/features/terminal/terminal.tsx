import { useTerminal } from "#/hooks/use-terminal";
import "@xterm/xterm/css/xterm.css";
import { RUNTIME_INACTIVE_STATES } from "#/types/agent-state";
import { cn } from "#/utils/utils";
import { WaitingForRuntimeMessage } from "../chat/waiting-for-runtime-message";
import { useAgentState } from "#/hooks/use-agent-state";
import { useElementDimensions } from "#/hooks/use-element-dimensions";
import React from "react";

function Terminal() {
  const { curAgentState } = useAgentState();

  const isRuntimeInactive = RUNTIME_INACTIVE_STATES.includes(curAgentState);

  const ref = useTerminal();
  const rootRef = React.useRef<HTMLDivElement>(null);
  
  // Debug: Monitor dimensions in real-time
  const dimensions = useElementDimensions(ref, !isRuntimeInactive);
  
  // #region agent log
  React.useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      const logData = {
        location: 'terminal.tsx:render',
        message: 'Terminal component render',
        data: {
          isRuntimeInactive,
          curAgentState,
          containerDimensions: dimensions,
          hasRef: !!ref.current,
          hasRootRef: !!rootRef.current,
          rootDimensions: rootRef.current ? {
            width: rootRef.current.clientWidth,
            height: rootRef.current.clientHeight,
            offsetParent: !!rootRef.current.offsetParent,
            computedStyle: {
              display: window.getComputedStyle(rootRef.current).display,
              position: window.getComputedStyle(rootRef.current).position,
            }
          } : null,
        },
        timestamp: Date.now(),
        sessionId: 'debug-session',
        runId: 'check-terminal',
        hypothesisId: 'H1'
      };
      fetch('http://localhost:42871/ingest/c1349573-fbfc-4b88-a75e-5f91da4b7b4b',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(logData)}).catch(()=>{});
      console.log('[Terminal] Render:', logData.data);
    }
  }, [isRuntimeInactive, curAgentState, dimensions, ref]);
  // #endregion

  return (
    <div 
      ref={rootRef}
      className="absolute inset-0 flex flex-col rounded-xl"
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
