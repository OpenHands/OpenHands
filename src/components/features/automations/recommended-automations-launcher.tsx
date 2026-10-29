import { useCallback, useMemo, useRef, useState } from "react";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useNavigation } from "#/context/navigation-context";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useSettings } from "#/hooks/query/use-settings";
import { useIsCreatingConversation } from "#/hooks/use-is-creating-conversation";
import { useResponderUrlSecret } from "#/hooks/use-responder-url-secret";
import { useConversationStore } from "#/stores/conversation-store";
import {
  setConversationState,
  setPendingTaskDraft,
} from "#/utils/conversation-local-storage";
import type { RecommendedAutomation } from "@openhands/extensions/automations";
import { parseMcpConfig } from "#/utils/mcp-config";
import { flattenMcpConfig } from "#/utils/mcp-installed-servers";
import {
  INTEGRATION_CATALOG as MCP_MARKETPLACE,
  type IntegrationCatalogEntry as MarketplaceEntry,
} from "@openhands/extensions/integrations";
import {
  findInstalledEntryMatch,
  getMarketplaceEntryById,
  isMcpInstallableEntry,
} from "#/utils/mcp-marketplace-utils";
import { InstallServerModal } from "#/components/features/mcp-page/install-server-modal";
import { useTracking } from "#/hooks/use-tracking";
import {
  automationSetupPath,
  hasAutomationInterface,
} from "#/manifests/automation-interface";
import { SETUP_REGISTRY } from "#/manifests/manifest-sources";
import {
  getAutomationLaunchPrompt,
  getHiddenAutomationIds,
  getRequiredIntegrationIds,
} from "#/utils/automation-catalog";
import { isResponderAutomation } from "#/utils/responder-deployment";
import { useAutomationHealth } from "#/hooks/query/use-automation-health";
import { useAutomations } from "#/hooks/query/use-automations";
import { useBuiltInIntegrationConnections } from "#/hooks/query/use-built-in-integration-connections";
import { useNativeGitIntegrations } from "#/hooks/query/use-native-git-integrations";
import { cloudIntegrationsUrl } from "#/utils/cloud-integrations-url";
import {
  BuiltInIntegrationChoiceModal,
  type BuiltInIntegrationState,
} from "./built-in-integration-choice-modal";
import { RecommendedAutomationsRail } from "./recommended-automations-rail";
import { RecommendedAutomationsSection } from "./recommended-automations-section";
import { ResponderDeploymentModal } from "./responder-deployment-modal";
import type { TemplateFilters } from "./template-filter-chips";

interface RecommendedAutomationsLauncherProps {
  query?: string;
  /** Resets the search from the catalog's no-match state. */
  onClearQuery?: () => void;
  /** The templates page's integration and trigger filters. */
  filters?: TemplateFilters;
  onFiltersChange?: (next: TemplateFilters) => void;
  onLaunched?: () => void;
  /** When true, only the automation card grid scrolls inside its section. */
  scrollableGrid?: boolean;
  /**
   * Compact discovery rail for New Chat and the automations dashboard.
   * The templates page keeps the full catalog section.
   */
  variant?: "catalog" | "rail";
  className?: string;
  dismissible?: boolean;
  onDismiss?: () => void;
}

/**
 * The marketplace entries a launch may wait on. An integration the automation
 * is willing to start without is deliberately absent, so it never queues an
 * install modal the user has to dismiss.
 */
function getRequiredEntries(automation: RecommendedAutomation) {
  return getRequiredIntegrationIds(automation)
    .map((id) => getMarketplaceEntryById(id, MCP_MARKETPLACE))
    .filter((entry): entry is MarketplaceEntry => !!entry);
}

/**
 * The automations a built-in integration (Settings > Integrations) already
 * provides out of the box — the 'Issue to PR' workflows and the Slack
 * responder — keyed to that integration's catalog entry.
 */
const BUILT_IN_INTEGRATION_ID_BY_AUTOMATION_ID: Record<string, string> = {
  "slack-channel-monitor": "slack",
  "github-issue-to-pr": "github",
  "gitlab-issue-to-mr": "gitlab",
  "jira-issue-to-pr": "jira",
  "jira-issue-to-gitlab-mr": "jira",
  "jira-issue-to-bitbucket-pr": "jira",
  "linear-issue-to-github-pr": "linear",
  "linear-issue-to-gitlab-mr": "linear",
  "linear-issue-to-bitbucket-pr": "linear",
};

export function RecommendedAutomationsLauncher({
  query,
  onClearQuery,
  filters,
  onFiltersChange,
  onLaunched,
  scrollableGrid = false,
  variant = "catalog",
  className,
  dismissible,
  onDismiss,
}: RecommendedAutomationsLauncherProps) {
  const activeBackend = useActiveBackend();
  const { navigate } = useNavigation();
  const { data: settings } = useSettings();
  const { trackPrebuiltAutomationEnabled } = useTracking();
  const createConversation = useCreateConversation();
  const ensureResponderUrlSecret = useResponderUrlSecret();
  const isCreatingConversation = useIsCreatingConversation();
  const setMessageToSend = useConversationStore(
    (state) => state.setMessageToSend,
  );
  const [pendingAutomation, setPendingAutomation] =
    useState<RecommendedAutomation | null>(null);
  const [deploymentChoiceAutomation, setDeploymentChoiceAutomation] =
    useState<RecommendedAutomation | null>(null);
  const [builtInChoice, setBuiltInChoice] = useState<{
    automation: RecommendedAutomation;
    integration: MarketplaceEntry;
    state: BuiltInIntegrationState;
  } | null>(null);
  const [installQueue, setInstallQueue] = useState<MarketplaceEntry[]>([]);
  const completedInstallRef = useRef(false);
  const launchInFlightRef = useRef(false);
  const localSetupInFlightRef = useRef(false);
  const [isPreparingLocalResponder, setIsPreparingLocalResponder] =
    useState(false);
  const isRail = variant === "rail";
  // Backends without an automation service (remote agent-servers such as a
  // DigitalOcean sandbox) 404 the list, which would surface as an error toast.
  const { data: automationHealth } = useAutomationHealth();
  const { data: automationsData, isLoading: isAutomationsLoading } =
    useAutomations({ enabled: isRail && automationHealth?.status === "ok" });
  const {
    getNativeIntegration,
    builtInIntegrations,
    isJiraEnabled,
    isLinearEnabled,
    isSlackEnabled,
    deploymentMode,
    isLoading: isNativeIntegrationsLoading,
  } = useNativeGitIntegrations();
  const hiddenAutomationIds = useMemo(
    () => getHiddenAutomationIds({ deploymentMode, builtInIntegrations }),
    [deploymentMode, builtInIntegrations],
  );
  const { isSlackConnected, isJiraConnected } =
    useBuiltInIntegrationConnections({
      slackEnabled: isSlackEnabled,
      jiraEnabled: isJiraEnabled,
    });

  const installedMcpConfig = useMemo(
    () =>
      flattenMcpConfig(
        settings?.mcp_config ??
          parseMcpConfig(settings?.agent_settings?.mcp_config),
      ).filter((server) => server.enabled !== false),
    [settings?.agent_settings?.mcp_config, settings?.mcp_config],
  );

  const launchAutomation = useCallback(
    (automation: RecommendedAutomation) => {
      if (
        launchInFlightRef.current ||
        createConversation.isPending ||
        isCreatingConversation
      ) {
        return;
      }
      launchInFlightRef.current = true;

      // An automation that ships a setup experience is configured from its own
      // form, so the answers are collected before anything is created. The rest
      // still hand a slash command to an agent to interpret.
      if (SETUP_REGISTRY.findById(automation.id)) {
        navigate?.(automationSetupPath(automation.id));
        onLaunched?.();
        return;
      }

      const prompt = getAutomationLaunchPrompt(automation);

      createConversation.mutate(
        {},
        {
          onSuccess: (conversation) => {
            trackPrebuiltAutomationEnabled({
              automationName: automation.name,
              automationCategory: automation.category,
            });
            if (
              conversation.conversation_id.startsWith("task-") &&
              conversation.task_id
            ) {
              setPendingTaskDraft(conversation.task_id, prompt);
            } else {
              setConversationState(conversation.conversation_id, {
                draftMessage: prompt,
              });
            }
            navigate?.(`/conversations/${conversation.conversation_id}`);
            onLaunched?.();
            window.setTimeout(() => setMessageToSend(prompt), 0);
          },
          onError: () => {
            launchInFlightRef.current = false;
          },
        },
      );
    },
    [
      activeBackend.backend.kind,
      createConversation,
      isCreatingConversation,
      navigate,
      onLaunched,
      setMessageToSend,
      trackPrebuiltAutomationEnabled,
    ],
  );

  // A required integration is satisfied by an installed MCP server or a
  // connected native integration. One this backend can connect neither way
  // (e.g. Jira's HTTP-only option) is excluded — the install queue can't do
  // anything with it — but the automation card keeps it visible and labels it
  // as needing external setup.
  const getMissingEntries = useCallback(
    (automation: RecommendedAutomation) =>
      getRequiredEntries(automation).filter((entry) => {
        if (findInstalledEntryMatch(entry, installedMcpConfig)) return false;
        const native = getNativeIntegration(entry.id);
        if (native) return !native.isConnected;
        return isMcpInstallableEntry(entry);
      }),
    [getNativeIntegration, installedMcpConfig],
  );

  const proceedWithLocalLaunch = (automation: RecommendedAutomation) => {
    const missingEntries = getMissingEntries(automation);
    if (missingEntries.length === 0) {
      launchAutomation(automation);
      return;
    }

    setPendingAutomation(automation);
    setInstallQueue(missingEntries);
  };

  const isBuiltInIntegrationEnabled = (entryId: string) => {
    if (entryId === "slack") return isSlackEnabled;
    if (entryId === "jira") return isJiraEnabled;
    if (entryId === "linear") return isLinearEnabled;
    return getNativeIntegration(entryId) !== null;
  };

  // Whether the current user has connected the built-in integration. Linear
  // has no connection lookup on the cloud backend, so it is never known to be
  // connected and the modal keeps offering to connect it.
  const isBuiltInIntegrationConnected = (entryId: string) => {
    if (entryId === "slack") return isSlackConnected;
    if (entryId === "jira") return isJiraConnected;
    if (entryId === "linear") return false;
    return getNativeIntegration(entryId)?.isConnected === true;
  };

  // The built-in integration that already gives the user this automation's
  // 'Issue to PR' workflow, when the cloud instance has it enabled.
  const getBuiltInIntegration = (automation: RecommendedAutomation) => {
    const entryId = BUILT_IN_INTEGRATION_ID_BY_AUTOMATION_ID[automation.id];
    if (!entryId || !isBuiltInIntegrationEnabled(entryId)) return null;
    return getMarketplaceEntryById(entryId, MCP_MARKETPLACE) ?? null;
  };

  const handleSelectAutomation = (automation: RecommendedAutomation) => {
    if (
      launchInFlightRef.current ||
      createConversation.isPending ||
      isCreatingConversation ||
      installQueue.length > 0 ||
      deploymentChoiceAutomation !== null ||
      builtInChoice !== null
    ) {
      return;
    }

    // GitHub/Slack responders poll continuously; let the user choose where the
    // responder runs before committing to the local setup flow. On a cloud
    // backend the responder already runs in the cloud, so there is no choice.
    if (
      activeBackend.backend.kind === "local" &&
      isResponderAutomation(automation)
    ) {
      setDeploymentChoiceAutomation(automation);
      return;
    }

    // A built-in integration already covers this workflow: show how to use it
    // when the user has connected it, or offer to connect it, with the polling
    // automation as the alternative either way.
    const builtInIntegration = getBuiltInIntegration(automation);
    if (builtInIntegration) {
      setBuiltInChoice({
        automation,
        integration: builtInIntegration,
        state: isBuiltInIntegrationConnected(builtInIntegration.id)
          ? "connected"
          : "connect",
      });
      return;
    }

    proceedWithLocalLaunch(automation);
  };

  const handleBuiltInChoiceClose = () => {
    setBuiltInChoice(null);
  };

  const handleUseBuiltInIntegration = () => {
    setBuiltInChoice(null);
    window.open(
      cloudIntegrationsUrl(activeBackend.backend, activeBackend.orgId),
      "_blank",
      "noopener,noreferrer",
    );
  };

  const handleContinuePollingSetup = () => {
    const automation = builtInChoice?.automation;
    setBuiltInChoice(null);
    if (automation) proceedWithLocalLaunch(automation);
  };

  const handleDeploymentContinueLocal = async () => {
    const automation = deploymentChoiceAutomation;
    if (!automation || localSetupInFlightRef.current) return;

    localSetupInFlightRef.current = true;
    setIsPreparingLocalResponder(true);

    try {
      const isSecretReady = await ensureResponderUrlSecret();
      if (!isSecretReady) return;

      setDeploymentChoiceAutomation(null);
      proceedWithLocalLaunch(automation);
    } finally {
      localSetupInFlightRef.current = false;
      setIsPreparingLocalResponder(false);
    }
  };

  const handleDeploymentOpenUrl = (url: string) => {
    setDeploymentChoiceAutomation(null);
    window.open(url, "_blank", "noopener,noreferrer");
  };

  const handleDeploymentClose = () => {
    setDeploymentChoiceAutomation(null);
  };

  const cancelInstallFlow = () => {
    if (completedInstallRef.current) {
      completedInstallRef.current = false;
      return;
    }
    setPendingAutomation(null);
    setInstallQueue([]);
  };

  const handleInstallSuccess = () => {
    completedInstallRef.current = true;

    setInstallQueue((currentQueue) => {
      const nextQueue = currentQueue.slice(1);

      if (nextQueue.length === 0) {
        const automation = pendingAutomation;
        window.setTimeout(() => {
          setPendingAutomation(null);
          if (automation) launchAutomation(automation);
        }, 0);
      }

      return nextQueue;
    });
  };

  const installEntry = installQueue[0] ?? null;

  // Like every automation surface, the launcher renders only behind the
  // interface-manifest gate; New Chat mounts it outside the gated routes.
  if (!hasAutomationInterface()) return null;

  if (isRail && isAutomationsLoading) return null;

  // Which integrations a card still needs depends on what the cloud instance
  // connects natively; a card shown before that is known could be launched
  // with the wrong install queue.
  if (isNativeIntegrationsLoading) return null;

  return (
    <>
      {isRail ? (
        <RecommendedAutomationsRail
          className={className}
          installedAutomations={automationsData?.automations ?? []}
          hiddenAutomationIds={hiddenAutomationIds}
          onSelect={handleSelectAutomation}
          dismissible={dismissible}
          onDismiss={onDismiss}
        />
      ) : (
        <RecommendedAutomationsSection
          backendKind={activeBackend.backend.kind}
          installedServers={installedMcpConfig}
          hiddenAutomationIds={hiddenAutomationIds}
          getNativeIntegration={getNativeIntegration}
          query={query}
          onClearQuery={onClearQuery}
          filters={filters}
          onFiltersChange={onFiltersChange}
          onSelect={handleSelectAutomation}
          scrollableGrid={scrollableGrid}
        />
      )}

      {installEntry && (
        <InstallServerModal
          key={installEntry.id}
          entry={installEntry}
          existingServers={installedMcpConfig}
          onClose={cancelInstallFlow}
          onSuccess={handleInstallSuccess}
          mcpOnly
        />
      )}

      {builtInChoice && (
        <BuiltInIntegrationChoiceModal
          integration={builtInChoice.integration}
          state={builtInChoice.state}
          onClose={handleBuiltInChoiceClose}
          onUseBuiltIn={handleUseBuiltInIntegration}
          onContinuePolling={handleContinuePollingSetup}
        />
      )}

      <ResponderDeploymentModal
        isOpen={deploymentChoiceAutomation !== null}
        isPending={isPreparingLocalResponder}
        onClose={handleDeploymentClose}
        onContinueLocal={handleDeploymentContinueLocal}
        onOpenUrl={handleDeploymentOpenUrl}
      />
    </>
  );
}
