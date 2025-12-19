import { lazy, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { ConversationLoading } from "../../conversation-loading";
import { I18nKey } from "#/i18n/declaration";
import { TabWrapper } from "./tab-wrapper";
import { TabContainer } from "./tab-container";
import { TabContentArea } from "./tab-content-area";
import { ConversationTabTitle } from "../conversation-tab-title";
import Terminal from "#/components/features/terminal/terminal";
import { useConversationStore } from "#/stores/conversation-store";
import { useConversationId } from "#/hooks/use-conversation-id";

// Lazy load all tab components
const EditorTab = lazy(() => import("#/routes/changes-tab"));
const BrowserTab = lazy(() => import("#/routes/browser-tab"));
const ServedTab = lazy(() => import("#/routes/served-tab"));
const VSCodeTab = lazy(() => import("#/routes/vscode-tab"));
const PlannerTab = lazy(() => import("#/routes/planner-tab"));

export function ConversationTabContent() {
  const { selectedTab, shouldShownAgentLoading } = useConversationStore();
  const { conversationId } = useConversationId();

  const { t } = useTranslation();

  // Determine which tab is active based on the current path
  const isEditorActive = selectedTab === "editor";
  const isBrowserActive = selectedTab === "browser";
  const isServedActive = selectedTab === "served";
  const isVSCodeActive = selectedTab === "vscode";
  const isTerminalActive = selectedTab === "terminal";
  const isPlannerActive = selectedTab === "planner";

  // Define tab configurations
  const tabs = [
    { key: "editor", component: EditorTab, isActive: isEditorActive },
    {
      key: "browser",
      component: BrowserTab,
      isActive: isBrowserActive,
    },
    { key: "served", component: ServedTab, isActive: isServedActive },
    { key: "vscode", component: VSCodeTab, isActive: isVSCodeActive },
    {
      key: "terminal",
      component: Terminal,
      isActive: isTerminalActive,
    },
    {
      key: "planner",
      component: PlannerTab,
      isActive: isPlannerActive,
    },
  ];

  const conversationTabTitle = useMemo(() => {
    if (isEditorActive) {
      return t(I18nKey.COMMON$CHANGES);
    }
    if (isBrowserActive) {
      return t(I18nKey.COMMON$BROWSER);
    }
    if (isServedActive) {
      return t(I18nKey.COMMON$APP);
    }
    if (isVSCodeActive) {
      return t(I18nKey.COMMON$CODE);
    }
    if (isTerminalActive) {
      return t(I18nKey.COMMON$TERMINAL);
    }
    if (isPlannerActive) {
      return t(I18nKey.COMMON$PLANNER);
    }
    return "";
  }, [
    isEditorActive,
    isBrowserActive,
    isServedActive,
    isVSCodeActive,
    isTerminalActive,
    isPlannerActive,
  ]);

  const conversationKey = useMemo(() => {
    if (isEditorActive) {
      return "editor";
    }
    if (isBrowserActive) {
      return "browser";
    }
    if (isServedActive) {
      return "served";
    }
    if (isVSCodeActive) {
      return "vscode";
    }
    if (isTerminalActive) {
      return "terminal";
    }
    if (isPlannerActive) {
      return "planner";
    }
    return "";
  }, [
    isEditorActive,
    isBrowserActive,
    isServedActive,
    isVSCodeActive,
    isTerminalActive,
    isPlannerActive,
  ]);

  if (shouldShownAgentLoading) {
    return <ConversationLoading />;
  }

  return (
    <TabContainer>
      <ConversationTabTitle
        title={conversationTabTitle}
        conversationKey={conversationKey}
      />
      <TabContentArea>
        {tabs.map(({ key, component: Component, isActive }) => (
          <TabWrapper
            // Force Terminal tab remount to reset XTerm buffer/state when conversationId changes
            key={key === "terminal" ? `${key}-${conversationId}` : key}
            isActive={isActive}
          >
            <Component />
          </TabWrapper>
        ))}
      </TabContentArea>
    </TabContainer>
  );
}
