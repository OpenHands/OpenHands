import { ReactNode } from "react";
import { CONVERSATION_TAB_PANEL_ID } from "../conversation-tab-ids";

interface TabContainerProps {
  children: ReactNode;
  /**
   * Accessible name: the name of the tab whose content is showing. Passed as
   * text rather than as an `aria-labelledby` reference because that tab's
   * button leaves the DOM when a narrow drawer pushes it into the overflow
   * menu, which would leave the panel unnamed.
   */
  label: string;
}

export function TabContainer({ children, label }: TabContainerProps) {
  return (
    <div
      id={CONVERSATION_TAB_PANEL_ID}
      role="tabpanel"
      aria-label={label}
      className="flex flex-col h-full w-full"
    >
      {children}
    </div>
  );
}
