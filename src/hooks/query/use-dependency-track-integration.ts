import { useMemo } from "react";
import { useSettings } from "#/hooks/query/use-settings";
import { useSearchSecrets } from "#/hooks/query/use-get-secrets";
import { useLocalWorkspaces } from "#/hooks/query/use-local-workspaces";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import type { DependencyTrackIntegrationConfig } from "#/types/integrations";
import {
  EMPTY_DEPENDENCY_TRACK_CONFIG,
  findWorkspaceIdForPath,
  getDependencyTrackConfigForWorkspace,
  resolveDependencyTrackSecretName,
} from "#/utils/dependency-track-workspace-config";

export type DependencyTrackIntegrationState = {
  workspaceId: string | null;
  config: DependencyTrackIntegrationConfig;
  apiKeyIsSet: boolean;
  isReady: boolean;
  isLoading: boolean;
  secretName: string | null;
};

export function useDependencyTrackIntegration(
  workspaceId: string | null | undefined,
): DependencyTrackIntegrationState {
  const { data: settings, isLoading: settingsLoading } = useSettings();
  const { data: secrets, isLoading: secretsLoading } = useSearchSecrets();

  return useMemo(() => {
    const id = workspaceId?.trim() || null;
    const config = getDependencyTrackConfigForWorkspace(
      settings?.integrations,
      id,
    );
    const secretName = id ? resolveDependencyTrackSecretName(config, id) : null;
    const apiKeyIsSet = Boolean(
      secretName && (secrets ?? []).some((s) => s.name === secretName),
    );
    const isReady =
      Boolean(id) &&
      config.enabled &&
      Boolean(config.baseUrl) &&
      Boolean(config.projectUuid) &&
      apiKeyIsSet;

    return {
      workspaceId: id,
      config: id ? config : { ...EMPTY_DEPENDENCY_TRACK_CONFIG },
      apiKeyIsSet,
      isReady,
      isLoading: settingsLoading || secretsLoading,
      secretName,
    };
  }, [workspaceId, settings, secrets, settingsLoading, secretsLoading]);
}

export function useConversationDependencyTrackIntegration(): DependencyTrackIntegrationState {
  const { data: conversation, isLoading: conversationLoading } =
    useActiveConversation();
  const { data: workspacesData, isLoading: workspacesLoading } =
    useLocalWorkspaces();

  const workspaceId = useMemo(() => {
    const path =
      conversation?.selected_workspace?.trim() ||
      conversation?.workspace?.working_dir?.trim() ||
      null;
    return findWorkspaceIdForPath(workspacesData?.workspaces ?? [], path);
  }, [conversation, workspacesData]);

  const state = useDependencyTrackIntegration(workspaceId);

  return {
    ...state,
    isLoading: state.isLoading || conversationLoading || workspacesLoading,
  };
}
