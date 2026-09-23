import React from "react";
import { createPortal } from "react-dom";
import { cn } from "#/utils/utils";
import {
  MODAL_PORTAL_HOST_ATTRIBUTE,
  ModalPortalHostContext,
} from "#/contexts/modal-portal-host-context";
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
  const [canUseDOM, setCanUseDOM] = React.useState(false);
  const [modalPortalHost, setModalPortalHost] =
    React.useState<HTMLDivElement | null>(null);
  const scopedStyle = React.useMemo(
    () =>
      ({
        ...styleOverrides,
        ...style,
      }) as React.CSSProperties,
    [style, styleOverrides],
  );
  const portalScopedStyle = React.useMemo(
    () =>
      ({
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
          {children}
        </div>
      </div>
      {canUseDOM &&
        createPortal(
          <div
            ref={setModalPortalHost}
            data-agent-server-ui=""
            data-color-theme={colorTheme}
            data-color-scheme={appearance}
            {...{ [MODAL_PORTAL_HOST_ATTRIBUTE]: "" }}
            className={cn(appearance, "text-foreground")}
            data-theme={appearance}
            style={portalScopedStyle}
          />,
          document.body,
        )}
    </ModalPortalHostContext.Provider>
  );
}
