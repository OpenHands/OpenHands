import React from "react";
import { createPortal } from "react-dom";
import { cn } from "#/utils/utils";
import {
  MODAL_PORTAL_HOST_ATTRIBUTE,
  ModalPortalHostContext,
} from "#/contexts/modal-portal-host-context";
import {
  AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES,
  AGENT_SERVER_UI_DEFAULT_THEME,
  type AgentServerUIStyleOverrides,
  type AgentServerUITheme,
} from "#/styles/agent-server-ui-style-scope";

export interface AgentServerUIRootProps extends Omit<
  React.HTMLAttributes<HTMLDivElement>,
  "style"
> {
  children: React.ReactNode;
  theme?: AgentServerUITheme;
  style?: React.CSSProperties;
  styleOverrides?: AgentServerUIStyleOverrides;
  contentClassName?: string;
}

export function AgentServerUIRoot({
  children,
  theme = AGENT_SERVER_UI_DEFAULT_THEME,
  className,
  style,
  styleOverrides,
  contentClassName,
  ...divProps
}: AgentServerUIRootProps) {
  const [canUseDOM, setCanUseDOM] = React.useState(false);
  const [modalPortalHost, setModalPortalHost] =
    React.useState<HTMLDivElement | null>(null);
  const scopedStyle = React.useMemo(
    () =>
      ({
        ...AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES,
        ...styleOverrides,
        ...style,
      }) as React.CSSProperties,
    [style, styleOverrides],
  );
  const portalScopedStyle = React.useMemo(
    () =>
      ({
        ...AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES,
        ...styleOverrides,
      }) as React.CSSProperties,
    [styleOverrides],
  );

  React.useEffect(() => {
    setCanUseDOM(true);
  }, []);

  return (
    <ModalPortalHostContext.Provider value={modalPortalHost}>
      <div
        data-agent-server-ui=""
        {...divProps}
        className={className}
        // CSS custom properties injected onto the scope root so descendants can resolve var(--oh-*)
        style={scopedStyle}
      >
        <div
          className={cn(theme, contentClassName, "text-foreground")}
          data-theme={theme}
        >
          {children}
        </div>
      </div>
      {canUseDOM &&
        createPortal(
          <div
            ref={setModalPortalHost}
            data-agent-server-ui=""
            {...{ [MODAL_PORTAL_HOST_ATTRIBUTE]: "" }}
            className={cn(theme, "text-foreground")}
            data-theme={theme}
            style={portalScopedStyle}
          />,
          document.body,
        )}
    </ModalPortalHostContext.Provider>
  );
}
