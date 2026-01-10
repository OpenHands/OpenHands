import { useWindowSize } from "@uidotdev/usehooks";
import { MobileLayout } from "./mobile-layout";
import { DesktopLayout } from "./desktop-layout";
import { useConversationStore } from "#/stores/conversation-store";

export function ConversationMain() {
  const { width } = useWindowSize();
  const { isRightPanelShown } = useConversationStore();

  if (width && width <= 1024) {
    return <MobileLayout isRightPanelShown={isRightPanelShown} />;
  }

  return <DesktopLayout isRightPanelShown={isRightPanelShown} />;
}
