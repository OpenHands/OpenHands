import { useCallback } from "react";
import { useIsMutating, useQueryClient } from "@tanstack/react-query";
import { WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME } from "#/api/agent-profiles-service/agent-profiles-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  llmProfileMirrored,
  updateAgentProfile,
} from "#/hooks/mutation/sync-agent-profiles";
import { useAgentProfiles } from "#/hooks/query/use-agent-profiles";
import type { ProfileInfo } from "@openhands/typescript-client";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useCanManageOrgProfiles } from "#/hooks/use-can-manage-org-profiles";
import { useSwitchLlmProfileAndLog } from "#/hooks/mutation/use-switch-llm-profile-and-log";
import { SWITCH_LLM_PROFILE_MUTATION_KEY } from "#/hooks/mutation/use-switch-llm-profile";
import { useModelStore } from "#/stores/model-store";

export interface ChatInputLlmProfileState {
  profiles: ProfileInfo[];
  /** The LLM profile the running conversation is currently using. */
  currentProfileName: string | null;
  /** That profile's model id (for the tooltip / subtitle). */
  currentProfileModel: string | null;
  isLoading: boolean;
  isSwitching: boolean;
  /**
   * Whether picking a profile here would actually land. False on a cloud
   * start-task route; the surfaces still name the active profile, they just
   * drop the selectable rows.
   */
  canSwitchProfile: boolean;
  /**
   * Switch the LLM profile: live via `/switch_profile` inside a conversation.
   * On the home page it becomes the active agent profile's LLM, which for a
   * local `default` means activating it (see {@link useSwitchLlmProfile}). A
   * cloud member's home pick is held for the next home launch instead.
   */
  selectProfile: (profileName: string) => void;
}

/**
 * Backs the OpenHands LLM-profile switcher pill, both on the home page (where
 * it shows and sets the LLM the active agent profile launches with) and
 * inside a conversation (live swap). The ACP analog is
 * {@link useChatInputModelState}. Mirrors the former SwitchProfileButton's
 * resolution priority so a switch is reflected instantly.
 */
export function useChatInputLlmProfileState(): ChatInputLlmProfileState {
  const { conversationId } = useOptionalConversationId();
  const { data: conversation } = useActiveConversation();
  const { data, isLoading } = useLlmProfiles();
  const queryClient = useQueryClient();
  const { backend } = useActiveBackend();
  const { data: agentProfiles } = useAgentProfiles({
    enabled: !conversationId,
  });
  const canManageOrgProfiles = useCanManageOrgProfiles();
  const { switchAndLog } = useSwitchLlmProfileAndLog();
  // Read the switch's pending state globally by mutation key: the pill button
  // and the menu that fires the switch are separate hook instances, so a
  // per-observer `isPending` would never light up on the button (#1571).
  const isSwitching =
    useIsMutating({ mutationKey: SWITCH_LLM_PROFILE_MUTATION_KEY }) > 0;
  // Written by useSwitchLlmProfile's onSuccess -> recordModelSwitchMessage ->
  // recordSwitch on a successful switch, so the label/check update before the
  // conversation refetch lands the new llm_model.
  const optimisticActiveProfile = useModelStore((s) =>
    conversationId ? s.activeProfileByConversation[conversationId] : undefined,
  );
  const homeLlmProfile = useModelStore((s) => s.homeLlmProfile);
  const setHomeLlmProfile = useModelStore((s) => s.setHomeLlmProfile);

  const profiles = data?.profiles ?? [];
  const conversationModel = conversation?.llm_model ?? null;
  // A cloud `task-…` route has no real conversation id yet, so the
  // per-conversation /switch_profile would 404.
  const isTaskRoute = conversationId?.startsWith("task-") ?? false;
  // A cloud member's home pick names the profile the home launch (and its
  // start-task route) runs, until that profile is gone.
  const homePick =
    (!conversationId || isTaskRoute) &&
    homeLlmProfile &&
    profiles.some((p) => p.name === homeLlmProfile)
      ? homeLlmProfile
      : null;

  // Resolution priority (mirrors the former SwitchProfileButton):
  //   1. Optimistic (just-clicked) — instant feedback before the refetch.
  //   2. Profile stamped on the conversation, validated against the live list
  //      so a since-deleted/renamed profile falls through.
  //   3. Profile whose model matches the running llm_model.
  //   4. Before the first message, a cloud member's home pick, else the
  //      active agent profile's own LLM, which the launch runs, else the
  //      user-level active_profile.
  const stampedProfile = conversation?.active_profile ?? null;
  const conversationProfile =
    stampedProfile && profiles.some((p) => p.name === stampedProfile)
      ? stampedProfile
      : null;
  const activeAgentProfile = agentProfiles?.profiles?.find(
    (profile) =>
      profile.id != null &&
      profile.id === agentProfiles.active_agent_profile_id,
  );
  const agentLlmProfile =
    activeAgentProfile?.agent_kind === "openhands" &&
    profiles.some((p) => p.name === activeAgentProfile.llm_profile_ref)
      ? activeAgentProfile.llm_profile_ref
      : null;
  // A local `default` runs the active LLM profile, so a home pick activates
  // it; any other profile keeps the pick as its own LLM.
  const ownsHomeLlm =
    !conversationId &&
    agentLlmProfile !== null &&
    !(
      backend.kind === "local" &&
      activeAgentProfile?.name === WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME
    );
  const currentProfileName =
    optimisticActiveProfile ??
    conversationProfile ??
    (conversationModel
      ? (profiles.find((p) => p.model === conversationModel)?.name ?? null)
      : (homePick ?? agentLlmProfile ?? data?.active_profile ?? null));
  const currentProfileModel =
    profiles.find((p) => p.name === currentProfileName)?.model ??
    conversationModel ??
    null;

  // On a start-task route the pill still names the active profile; only the
  // selectable rows drop out.
  const canSwitchProfile = !isTaskRoute;

  const selectProfile = useCallback(
    (profileName: string) => {
      if (!canSwitchProfile || profileName === currentProfileName) return;
      if (!conversationId) {
        // A home-page pick is an explicit model choice made before the first
        // message: mark it so the next home launch suppresses "Run on first
        // message" routing for that one conversation. Set here — the single
        // chokepoint — so every pick path is covered: a cloud member's held
        // pick, an agent-profile-owned LLM pick, and a global activate all
        // flow through here. `useCreateConversation` reads-and-clears it.
        useModelStore.getState().setHomeManualModelPick(true);
        // A home pick edits the org's agent profile or activates the LLM
        // profile org-wide, both of which a cloud member can only 403 on.
        // Hold a member's pick for the next home launch (sent as
        // `llm_profile`).
        if (!canManageOrgProfiles) {
          setHomeLlmProfile(profileName);
          return;
        }
        setHomeLlmProfile(null);
        if (ownsHomeLlm && activeAgentProfile) {
          updateAgentProfile(
            queryClient,
            activeAgentProfile.name,
            llmProfileMirrored(profileName),
          );
          return;
        }
      }
      switchAndLog(conversationId, profileName);
    },
    [
      canManageOrgProfiles,
      canSwitchProfile,
      conversationId,
      currentProfileName,
      setHomeLlmProfile,
      switchAndLog,
      ownsHomeLlm,
      activeAgentProfile,
      queryClient,
    ],
  );

  return {
    profiles,
    currentProfileName,
    currentProfileModel,
    isLoading,
    isSwitching,
    canSwitchProfile,
    selectProfile,
  };
}
