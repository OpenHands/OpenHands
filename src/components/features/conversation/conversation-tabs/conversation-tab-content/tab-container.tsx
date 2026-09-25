import { ReactNode } from "react";
import { CONVERSATION_TAB_PANEL_ID } from "../conversation-tab-ids";

interface TabContainerProps {
  children: ReactNode;
  /** Id of the tab that opened this panel, for `aria-labelledby`. */
  labelledBy?: string;
}

export function TabContainer({ children, labelledBy }: TabContainerProps) {
  return (
    <div
      id={CONVERSATION_TAB_PANEL_ID}
      role="tabpanel"
      aria-labelledby={labelledBy}
      className="flex flex-col h-full w-full"
    >
      {children}
    </div>
  );
}
