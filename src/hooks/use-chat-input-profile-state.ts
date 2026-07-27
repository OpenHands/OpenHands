import { useCallback } from "react";
import { useIsMutating } from "@tanstack/react-query";
import type { AgentProfileSummary } from "#/api/agent-profiles-service/agent-profiles-service.api";
import { getStoredConversationMetadata } from "#/api/conversation-metadata-store";
import { useNavigation } from "#/context/navigation-context";
import { useAgentProfiles } from "#/hooks/query/use-agent-profiles";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import {
  CREATE_CONVERSATION_MUTATION_KEY,
  useCreateConversation,
  type CreateConversationVariables,
} from "#/hooks/mutation/use-create-conversation";
import {
  useActivateAgentProfile,
  ACTIVATE_AGENT_PROFILE_MUTATION_KEY,
} from "#/hooks/mutation/use-activate-agent-profile";

export interface ChatInputProfileState {
  profiles: AgentProfileSummary[];
  currentProfileId: string | null;
  currentProfileName: string | null;
  isInConversation: boolean;
  isLoading: boolean;
  isSwitching: boolean;
  selectProfile: (profile: AgentProfileSummary) => void;
}

export function useChatInputProfileState(): ChatInputProfileState {
  const { conversationId, navigate } = useNavigation();
  const { data: conversation, isLoading: isLoadingConversation } =
    useActiveConversation();
  const { data: agentProfiles, isLoading: isLoadingProfiles } =
    useAgentProfiles();
  const createConversation = useCreateConversation();
  const activateProfile = useActivateAgentProfile();
  const activatingProfiles = useIsMutating({
    mutationKey: ACTIVATE_AGENT_PROFILE_MUTATION_KEY,
  });
  const creatingConversations = useIsMutating({
    mutationKey: CREATE_CONVERSATION_MUTATION_KEY,
  });
  const isTaskRoute = conversationId?.startsWith("task-") ?? false;
  const isSwitching =
    isTaskRoute || activatingProfiles > 0 || creatingConversations > 0;

  const profiles = agentProfiles?.profiles ?? [];
  const isInConversation = Boolean(conversationId);
  const currentProfileId = isInConversation
    ? (conversation?.launched_agent_profile?.agent_profile_id ??
      agentProfiles?.active_agent_profile_id ??
      null)
    : (agentProfiles?.active_agent_profile_id ?? null);
  const currentProfileName =
    profiles.find((p) => p.id != null && p.id === currentProfileId)?.name ??
    null;

  const selectProfile = useCallback(
    (profile: AgentProfileSummary) => {
      if (isTaskRoute || !profile.id || profile.id === currentProfileId) return;
      if (!isInConversation) {
        activateProfile.mutate(profile.id);
        return;
      }

      const metadata = conversationId
        ? getStoredConversationMetadata(conversationId)
        : null;
      const variables: CreateConversationVariables = {
        agentProfileId: profile.id,
        entryPoint: "blank_conversation_profile_picker",
      };
      if (conversation?.selected_repository) {
        variables.repository = {
          name: conversation.selected_repository,
          gitProvider: conversation.git_provider ?? "github",
          branch: conversation.selected_branch ?? undefined,
        };
      }
      if (conversation?.selected_workspace) {
        variables.workingDir = conversation.selected_workspace;
        variables.workspaceMode = metadata?.workspace_mode ?? "local_repo";
      }
      if (metadata?.plugins?.length) {
        variables.plugins = metadata.plugins;
      }

      createConversation.mutate(variables, {
        onSuccess: (data) => navigate(`/conversations/${data.conversation_id}`),
      });
    },
    [
      activateProfile,
      conversation,
      conversationId,
      createConversation,
      currentProfileId,
      isInConversation,
      isTaskRoute,
      navigate,
    ],
  );

  return {
    profiles,
    currentProfileId,
    currentProfileName,
    isInConversation,
    isLoading:
      isLoadingProfiles ||
      isTaskRoute ||
      (isInConversation && isLoadingConversation),
    isSwitching,
    selectProfile,
  };
}
