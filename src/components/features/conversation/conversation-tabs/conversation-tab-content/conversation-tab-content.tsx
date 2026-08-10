import { lazy, useMemo } from "react";
import { TabWrapper } from "./tab-wrapper";
import { TabContainer } from "./tab-container";
import { TabContentArea } from "./tab-content-area";
import { ConversationTabContentCrossfade } from "./conversation-tab-content-crossfade";
import { useConversationStore } from "#/stores/conversation-store";
import { useConversationId } from "#/hooks/use-conversation-id";

// Lazy load all tab components, including the terminal — xterm + addon-fit +
// xterm.css are large enough that we don't want them in the conversation
// route's eager graph just because the terminal tab might be selected later.
const FilesTab = lazy(() => import("#/routes/files-tab"));
const BrowserTab = lazy(() => import("#/routes/browser-tab"));
const PlannerTab = lazy(() => import("#/routes/planner-tab"));
const TaskListTab = lazy(() => import("#/routes/task-list-tab"));
const CloudAiTab = lazy(() => import("#/routes/cloudai-tab"));
const SecurityTab = lazy(() => import("#/routes/security-tab"));
const DesktopTab = lazy(() => import("#/routes/desktop-tab"));
const EmulatorTab = lazy(() => import("#/routes/emulator-tab"));
const Terminal = lazy(() => import("#/components/features/terminal/terminal"));

const TAB_CONFIG = {
  tasklist: { component: TaskListTab },
  files: { component: FilesTab },
  browser: { component: BrowserTab },
  terminal: { component: Terminal },
  planner: { component: PlannerTab },
  cloudai: { component: CloudAiTab },
  security: { component: SecurityTab },
  desktop: { component: DesktopTab },
  emulator: { component: EmulatorTab },
};

export function ConversationTabContent() {
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
    <TabContainer>
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
