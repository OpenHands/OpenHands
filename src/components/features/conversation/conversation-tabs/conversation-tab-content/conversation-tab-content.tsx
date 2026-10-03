import { lazy, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { TabWrapper } from "./tab-wrapper";
import { TabContainer } from "./tab-container";
import { TabContentArea } from "./tab-content-area";
import { ConversationTabContentCrossfade } from "./conversation-tab-content-crossfade";
import { I18nKey } from "#/i18n/declaration";
import { useConversationStore } from "#/stores/conversation-store";
import { useConversationId } from "#/hooks/use-conversation-id";

// Lazy load all tab components, including the terminal — xterm + addon-fit +
// xterm.css are large enough that we don't want them in the conversation
// route's eager graph just because the terminal tab might be selected later.
const FilesTab = lazy(() => import("#/routes/files-tab"));
const CommitsTab = lazy(() => import("#/routes/commits-tab"));
const BrowserTab = lazy(() => import("#/routes/browser-tab"));
const PlannerTab = lazy(() => import("#/routes/planner-tab"));
const TaskListTab = lazy(() => import("#/routes/task-list-tab"));
const UsageTab = lazy(() => import("#/routes/usage-tab"));
const Terminal = lazy(() => import("#/components/features/terminal/terminal"));

// `labelKey` matches the tab's name in the strip; it names the panel.
const TAB_CONFIG = {
  tasklist: { component: TaskListTab, labelKey: I18nKey.COMMON$TASK_LIST },
  files: { component: FilesTab, labelKey: I18nKey.COMMON$FILES },
  commits: { component: CommitsTab, labelKey: I18nKey.DIFF_VIEWER$COMMITS },
  browser: { component: BrowserTab, labelKey: I18nKey.COMMON$BROWSER },
  terminal: { component: Terminal, labelKey: I18nKey.COMMON$TERMINAL },
  planner: { component: PlannerTab, labelKey: I18nKey.COMMON$PLANNER },
  usage: { component: UsageTab, labelKey: I18nKey.COMMON$USAGE },
};

export function ConversationTabContent() {
  const { t } = useTranslation("openhands");
  const { selectedTab, shouldShownAgentLoading } = useConversationStore();
  const { conversationId } = useConversationId();

  const activeTab = useMemo(
    () =>
      TAB_CONFIG[selectedTab as keyof typeof TAB_CONFIG] ?? TAB_CONFIG.files,
    [selectedTab],
  );

  const ActiveComponent = activeTab.component;

  const tabWrapperKey =
    selectedTab === "terminal"
      ? `${selectedTab}-${conversationId}`
      : (selectedTab ?? "files");

  return (
    <TabContainer label={t(activeTab.labelKey)}>
      <TabContentArea>
        <ConversationTabContentCrossfade
          showAgentLoading={shouldShownAgentLoading}
          tabKey={tabWrapperKey}
        >
          <TabWrapper key={tabWrapperKey}>
            <ActiveComponent />
          </TabWrapper>
        </ConversationTabContentCrossfade>
      </TabContentArea>
    </TabContainer>
  );
}
