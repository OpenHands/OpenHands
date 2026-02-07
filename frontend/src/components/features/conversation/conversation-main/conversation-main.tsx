import React from "react";
import { MobileLayout } from "./mobile-layout";
import { DesktopLayout } from "./desktop-layout";
import { useConversationStore } from "#/stores/conversation-store";

const mql = window.matchMedia("(max-width: 1024px)");

function subscribe(callback: () => void) {
  mql.addEventListener("change", callback);
  return () => mql.removeEventListener("change", callback);
}

function getSnapshot() {
  return mql.matches;
}

export function ConversationMain() {
  const isMobile = React.useSyncExternalStore(subscribe, getSnapshot);
  const { isRightPanelShown } = useConversationStore();

  if (isMobile) {
    return <MobileLayout isRightPanelShown={isRightPanelShown} />;
  }

  return <DesktopLayout isRightPanelShown={isRightPanelShown} />;
}
