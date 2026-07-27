import { useCallback, useMemo, useRef, useState } from "react";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useNavigation } from "#/context/navigation-context";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useSettings } from "#/hooks/query/use-settings";
import { useIsCreatingConversation } from "#/hooks/use-is-creating-conversation";
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
  getMcpMarketplaceCatalog,
} from "#/utils/mcp-marketplace-utils";
import { InstallServerModal } from "#/components/features/mcp-page/install-server-modal";
import { useTracking } from "#/hooks/use-tracking";
import { isResponderAutomation } from "#/utils/responder-deployment";
import { RecommendedAutomationsSection } from "./recommended-automations-section";
import { ResponderDeploymentModal } from "./responder-deployment-modal";

interface RecommendedAutomationsLauncherProps {
  query?: string;
  onLaunched?: () => void;
  /** When true, only the automation card grid scrolls inside its section. */
  scrollableGrid?: boolean;
}

function getRequiredEntries(automation: RecommendedAutomation) {
  const mcpMarketplace = getMcpMarketplaceCatalog(MCP_MARKETPLACE);
  return automation.requiredIntegrationIds
    .map((id) => getMarketplaceEntryById(id, mcpMarketplace))
    .filter((entry): entry is MarketplaceEntry => !!entry);
}

/**
 * The catalog prompt (or slash command) is passed through as-is.
 * API routing (host, auth) is discovered by the agent at runtime from
 * `<RUNTIME_SERVICES>` in the system prompt — the skills themselves
 * contain the instructions for reading that block.
 */
export function buildAutomationPrompt(basePrompt: string): string {
  return basePrompt;
}

export function RecommendedAutomationsLauncher({
  query,
  onLaunched,
  scrollableGrid = false,
}: RecommendedAutomationsLauncherProps) {
  const activeBackend = useActiveBackend();
  const { navigate } = useNavigation();
  const { data: settings } = useSettings();
  const { trackPrebuiltAutomationEnabled } = useTracking();
  const createConversation = useCreateConversation();
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

  const installedMcpConfig = useMemo(
    () =>
      flattenMcpConfig(parseMcpConfig(settings?.agent_settings?.mcp_config)),
    [settings?.agent_settings?.mcp_config],
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

      const prompt = buildAutomationPrompt(automation.prompt);

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

  const handleDeploymentContinueLocal = () => {
    const automation = deploymentChoiceAutomation;
    setDeploymentChoiceAutomation(null);
    if (automation) {
      proceedWithLocalLaunch(automation);
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

  // Recommended automations are a local-backend-only feature; cloud
  // automations are managed elsewhere.
  if (activeBackend.backend.kind === "cloud") return null;

  return (
    <>
      <RecommendedAutomationsSection
        backendKind={activeBackend.backend.kind}
        installedServers={installedMcpConfig}
        query={query}
        onSelect={handleSelectAutomation}
        scrollableGrid={scrollableGrid}
      />

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
        onClose={handleDeploymentClose}
        onContinueLocal={handleDeploymentContinueLocal}
        onOpenUrl={handleDeploymentOpenUrl}
      />
    </>
  );
}
