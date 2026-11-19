import type { ReactNode } from "react";
import { createPortal } from "react-dom";
import { useAgentServerUIAppearance } from "#/components/providers/agent-server-ui-root";

interface AppearancePortalProps {
  children: ReactNode;
  container?: Element | DocumentFragment;
}

/**
 * Portals outside AgentServerUIRoot's content wrapper lose its appearance
 * class, so `dark:` utilities stop matching. The `contents` wrapper restores
 * the same class/data-theme pair without adding a layout box.
 */
export function AppearancePortal({
  children,
  container,
}: AppearancePortalProps) {
  const appearance = useAgentServerUIAppearance();
  if (typeof document === "undefined") return null;
  return createPortal(
    <div className={`${appearance} contents`} data-theme={appearance}>
      {children}
    </div>,
    container ?? document.body,
  );
}
