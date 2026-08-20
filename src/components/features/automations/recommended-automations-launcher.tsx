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
  getRequiredIntegrationIds,
} from "#/utils/automation-catalog";
import { isResponderAutomation } from "#/utils/responder-deployment";
import { useAutomations } from "#/hooks/query/use-automations";
import { RecommendedAutomationsRail } from "./recommended-automations-rail";
import { RecommendedAutomationsSection } from "./recommended-automations-section";
import { ResponderDeploymentModal } from "./responder-deployment-modal";

interface RecommendedAutomationsLauncherProps {
  query?: string;
  onLaunched?: () => void;
  /** When true, only the automation card grid scrolls inside its section. */
  scrollableGrid?: boolean;
  /**
   * Compact discovery rail for New Chat and the automations dashboard.
   * The templates page keeps the full catalog section.
   */
  variant?: "catalog" | "rail";
  className?: string;
}

/**
 * The marketplace entries a launch waits on. An integration the automation is
 * willing to start without is deliberately absent, so it never queues an
 * install modal the user has to dismiss.
 *
 * A required integration this backend cannot install as MCP (e.g. Jira's
 * HTTP-only option) is also excluded here — the install queue can't do
 * anything with it — but it is not silently dropped from the product: the
 * automation card keeps it visible and labels it as needing external setup.
 */
function getRequiredEntries(automation: RecommendedAutomation) {
  return getRequiredIntegrationIds(automation)
    .map((id) => getMarketplaceEntryById(id, MCP_MARKETPLACE))
    .filter((entry): entry is MarketplaceEntry => !!entry)
    .filter(isMcpInstallableEntry);
}

export function RecommendedAutomationsLauncher({
  query,
  onLaunched,
  scrollableGrid = false,
  variant = "catalog",
  className,
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
  const [installQueue, setInstallQueue] = useState<MarketplaceEntry[]>([]);
  const completedInstallRef = useRef(false);
  const launchInFlightRef = useRef(false);
  const localSetupInFlightRef = useRef(false);
  const [isPreparingLocalResponder, setIsPreparingLocalResponder] =
    useState(false);
  const isRail = variant === "rail";
  const { data: automationsData, isLoading: isAutomationsLoading } =
    useAutomations({
      enabled: isRail && activeBackend.backend.kind === "local",
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

  const getMissingEntries = useCallback(
    (automation: RecommendedAutomation) =>
      getRequiredEntries(automation).filter(
        (entry) => !findInstalledEntryMatch(entry, installedMcpConfig),
      ),
    [installedMcpConfig],
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

  const handleSelectAutomation = (automation: RecommendedAutomation) => {
    if (
      launchInFlightRef.current ||
      createConversation.isPending ||
      isCreatingConversation ||
      installQueue.length > 0 ||
      deploymentChoiceAutomation !== null
    ) {
      return;
    }

    // GitHub/Slack responders poll continuously; let the user choose where the
    // responder runs before committing to the local setup flow.
    if (isResponderAutomation(automation)) {
      setDeploymentChoiceAutomation(automation);
      return;
    }

    proceedWithLocalLaunch(automation);
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

  // Recommended automations are a local-backend-only feature; cloud
  // automations are managed elsewhere.
  if (activeBackend.backend.kind === "cloud") return null;

  if (isRail && isAutomationsLoading) return null;

  return (
    <>
      {isRail ? (
        <RecommendedAutomationsRail
          className={className}
          installedAutomations={automationsData?.automations ?? []}
          onSelect={handleSelectAutomation}
        />
      ) : (
        <RecommendedAutomationsSection
          backendKind={activeBackend.backend.kind}
          installedServers={installedMcpConfig}
          query={query}
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
