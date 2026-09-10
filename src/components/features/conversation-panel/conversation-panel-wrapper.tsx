import ReactDOM from "react-dom";
import { useNavigation } from "#/contexts/navigation-context";
import { cn } from "#/utils/utils";

interface ConversationPanelWrapperProps {
  isOpen: boolean;
}

export function ConversationPanelWrapper({
  isOpen,
  children,
}: React.PropsWithChildren<ConversationPanelWrapperProps>) {
  const { currentPath } = useNavigation();

  if (!isOpen) return null;

  const portalTarget = document.getElementById("root-outlet");
  if (!portalTarget) return null;

  return ReactDOM.createPortal(
    <div
      className={cn(
        "absolute h-full w-full left-0 top-0 z-[100] bg-black/80 rounded-xl",
        currentPath === "/" && "bottom-0 top-0 md:top-3 md:bottom-3 h-auto",
      )}
    >
      {children}
    </div>,
    portalTarget,
  );
}
