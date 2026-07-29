import React from "react";
import { cn } from "#/utils/utils";
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
  const scopedStyle = React.useMemo(
    () =>
      ({
        ...AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES,
        ...styleOverrides,
        ...style,
      }) as React.CSSProperties,
    [style, styleOverrides],
  );

  return (
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
  );
}
