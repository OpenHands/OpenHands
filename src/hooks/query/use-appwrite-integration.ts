import { useMemo } from "react";
import { useSettings } from "#/hooks/query/use-settings";
import { useSearchSecrets } from "#/hooks/query/use-get-secrets";
import { useLocalWorkspaces } from "#/hooks/query/use-local-workspaces";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import type { AppwriteIntegrationConfig } from "#/types/integrations";
import {
  EMPTY_APPWRITE_CONFIG,
  findWorkspaceIdForPath,
  getAppwriteConfigForWorkspace,
  resolveAppwriteSecretName,
} from "#/utils/appwrite-workspace-config";

export type AppwriteIntegrationState = {
  workspaceId: string | null;
  config: AppwriteIntegrationConfig;
  apiKeyIsSet: boolean;
  /** True when a workspace is selected and its AppWrite config is complete. */
  isReady: boolean;
  isLoading: boolean;
  secretName: string | null;
};

/**
 * Reads AppWrite integration config for a specific workspace id.
 */
export function useAppwriteIntegration(
  workspaceId: string | null | undefined,
): AppwriteIntegrationState {
  const { data: settings, isLoading: settingsLoading } = useSettings();
  const { data: secrets, isLoading: secretsLoading } = useSearchSecrets();

  return useMemo(() => {
    const id = workspaceId?.trim() || null;
    const config = getAppwriteConfigForWorkspace(settings?.integrations, id);
    const secretName = id ? resolveAppwriteSecretName(config, id) : null;
    const apiKeyIsSet = Boolean(
      secretName && (secrets ?? []).some((s) => s.name === secretName),
    );
    const isReady =
      Boolean(id) &&
      config.enabled &&
      Boolean(config.endpoint) &&
      Boolean(config.projectId) &&
      apiKeyIsSet;

    return {
      workspaceId: id,
      config: id ? config : { ...EMPTY_APPWRITE_CONFIG },
      apiKeyIsSet,
      isReady,
      isLoading: settingsLoading || secretsLoading,
      secretName,
    };
  }, [workspaceId, settings, secrets, settingsLoading, secretsLoading]);
}

/**
 * Resolves the conversation's attached workspace and returns its AppWrite
 * integration state (used by the CloudAI tab gate).
 */
export function useConversationAppwriteIntegration(): AppwriteIntegrationState {
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

  const state = useAppwriteIntegration(workspaceId);

  return {
    ...state,
    isLoading: state.isLoading || conversationLoading || workspacesLoading,
  };
}
