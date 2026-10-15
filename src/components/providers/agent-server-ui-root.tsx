import React from "react";
import { cn } from "#/utils/utils";
import {
  type AgentServerUIStyleOverrides,
  type AgentServerUITheme,
} from "#/styles/agent-server-ui-style-scope";
import { useColorTheme } from "#/hooks/use-color-theme";
import { COLOR_THEMES } from "#/themes/color-themes";

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

const AgentServerUIAppearanceContext =
  React.createContext<AgentServerUITheme | null>(null);

/** The appearance of the nearest AgentServerUIRoot (or the active color theme). */
export function useAgentServerUIAppearance(): AgentServerUITheme {
  const colorTheme = useColorTheme();
  return (
    React.useContext(AgentServerUIAppearanceContext) ??
    COLOR_THEMES[colorTheme].appearance
  );
}

export function AgentServerUIRoot({
  children,
  theme,
  className,
  style,
  styleOverrides,
  contentClassName,
  ...divProps
}: AgentServerUIRootProps) {
  const colorTheme = useColorTheme();
  const appearance = theme ?? COLOR_THEMES[colorTheme].appearance;
  const scopedStyle = React.useMemo(
    () =>
      ({
        ...styleOverrides,
        ...style,
      }) as React.CSSProperties,
    [style, styleOverrides],
  );

  return (
    <div
      data-agent-server-ui=""
      data-color-theme={colorTheme}
      data-color-scheme={appearance}
      {...divProps}
      className={className}
      // Only consumer overrides are inline; theme defaults belong to CSS.
      style={scopedStyle}
    >
      <div
        className={cn(appearance, contentClassName, "text-foreground")}
        data-theme={appearance}
      >
        <AgentServerUIAppearanceContext.Provider value={appearance}>
          {children}
        </AgentServerUIAppearanceContext.Provider>
      </div>
    </div>
  );
}
