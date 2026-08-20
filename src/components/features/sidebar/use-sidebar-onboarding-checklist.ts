import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { useOnboardingCompletion } from "#/components/features/onboarding/use-onboarding-completion";
import { useNavigation } from "#/context/navigation-context";
import { useAutomations } from "#/hooks/query/use-automations";
import { useAutomationHealth } from "#/hooks/query/use-automation-health";
import { usePaginatedConversations } from "#/hooks/query/use-paginated-conversations";
import { useSettings } from "#/hooks/query/use-settings";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useLlmConfigured } from "#/hooks/use-llm-configured";
import { parseMcpConfig } from "#/utils/mcp-config";
import {
  isCustomizeChecklistPath,
  SIDEBAR_ONBOARDING_CHECKLIST_ITEM_IDS,
  type SidebarOnboardingChecklistItemId,
} from "./sidebar-onboarding-checklist.constants";
import { isConfigureLlmChecklistItemComplete } from "./sidebar-onboarding-checklist-llm-complete";
import {
  readSidebarOnboardingChecklistCustomizeExplored,
  readSidebarOnboardingChecklistMinimized,
  readSidebarOnboardingChecklistSlackJoined,
  subscribeSidebarOnboardingChecklistDismissed,
  getSidebarOnboardingChecklistDismissedSnapshot,
  writeSidebarOnboardingChecklistCustomizeExplored,
  writeSidebarOnboardingChecklistDismissed,
  writeSidebarOnboardingChecklistMinimized,
  writeSidebarOnboardingChecklistSlackJoined,
} from "./sidebar-onboarding-checklist-storage";

export interface SidebarOnboardingChecklistItemState {
  id: SidebarOnboardingChecklistItemId;
  isComplete: boolean;
}

function hasConfiguredMcpServers(mcpConfig: unknown): boolean {
  return Object.keys(parseMcpConfig(mcpConfig)).length > 0;
}

export function useSidebarOnboardingChecklist() {
  const { isCompleted: onboardingCompleted } = useOnboardingCompletion();
  const { currentPath } = useNavigation();
  const isDismissed = useSyncExternalStore(
    subscribeSidebarOnboardingChecklistDismissed,
    getSidebarOnboardingChecklistDismissedSnapshot,
    () => false,
  );
  const [isMinimized, setIsMinimized] = useState(
    readSidebarOnboardingChecklistMinimized,
  );
  const [hasExploredCustomize, setHasExploredCustomize] = useState(
    readSidebarOnboardingChecklistCustomizeExplored,
  );
  const [hasJoinedSlack, setHasJoinedSlack] = useState(
    readSidebarOnboardingChecklistSlackJoined,
  );

  const { data: settings } = useSettings();
  const { isConfigured: isLlmConfigured, isLoading: isLlmConfiguredLoading } =
    useLlmConfigured();
  const { data: profilesData, isLoading: isProfilesLoading } = useLlmProfiles();
  const { data: conversationPage } = usePaginatedConversations(1);
  const { data: healthData } = useAutomationHealth();
  const isAutomationBackendHealthy = healthData?.status === "ok";
  const { data: automationsData } = useAutomations({
    limit: 1,
    offset: 0,
    enabled: isAutomationBackendHealthy,
  });

  useEffect(() => {
    if (!isCustomizeChecklistPath(currentPath)) {
      return;
    }

    if (hasExploredCustomize) {
      return;
    }

    writeSidebarOnboardingChecklistCustomizeExplored(true);
    setHasExploredCustomize(true);
  }, [currentPath, hasExploredCustomize]);

  const completionById = useMemo(() => {
    const hasConversation = (conversationPage?.pages[0]?.items.length ?? 0) > 0;
    const hasAutomation = (automationsData?.total ?? 0) > 0;

    return {
      "configure-llm": isConfigureLlmChecklistItemComplete(
        settings,
        isLlmConfigured,
        isLlmConfiguredLoading,
        profilesData,
        isProfilesLoading,
      ),
      "connect-mcp": hasConfiguredMcpServers(
        settings?.agent_settings?.mcp_config,
      ),
      "start-conversation": hasConversation,
      "schedule-task": hasAutomation,
      "customize-agent": hasExploredCustomize,
      "join-slack": hasJoinedSlack,
    } satisfies Record<SidebarOnboardingChecklistItemId, boolean>;
  }, [
    automationsData?.total,
    conversationPage?.pages,
    hasExploredCustomize,
    hasJoinedSlack,
    isLlmConfigured,
    isLlmConfiguredLoading,
    isProfilesLoading,
    profilesData,
    settings,
    settings?.agent_settings?.mcp_config,
  ]);

  const items = useMemo(
    (): SidebarOnboardingChecklistItemState[] =>
      SIDEBAR_ONBOARDING_CHECKLIST_ITEM_IDS.map((id) => ({
        id,
        isComplete: completionById[id],
      })),
    [completionById],
  );

  const completedCount = items.filter((item) => item.isComplete).length;
  const isAllComplete = completedCount === items.length;

  const isVisible = onboardingCompleted && !isDismissed && !isAllComplete;

  const dismiss = () => {
    writeSidebarOnboardingChecklistDismissed(true);
  };

  const toggleMinimized = () => {
    setIsMinimized((current) => {
      const next = !current;
      writeSidebarOnboardingChecklistMinimized(next);
      return next;
    });
  };

  const markJoinSlackComplete = () => {
    if (hasJoinedSlack) {
      return;
    }
    writeSidebarOnboardingChecklistSlackJoined(true);
    setHasJoinedSlack(true);
  };

  return {
    items,
    completedCount,
    totalCount: items.length,
    isVisible,
    isMinimized,
    dismiss,
    toggleMinimized,
    markJoinSlackComplete,
  };
}
