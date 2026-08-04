import { useCallback } from "react";
import { useIsMutating } from "@tanstack/react-query";
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
   * start-task route and for a cloud member on the home page; the surfaces
   * still name the active profile, they just drop the selectable rows.
   */
  canSwitchProfile: boolean;
  /**
   * Switch the LLM profile: live via `/switch_profile` when inside a
   * conversation, or activate it globally when on the home page (no
   * conversation) — see {@link useSwitchLlmProfile}.
   */
  selectProfile: (profileName: string) => void;
}

/**
 * Backs the OpenHands LLM-profile switcher pill, both on the home page (where
 * a pick activates the profile globally so the next conversation launches
 * with it) and inside a conversation (live swap). The ACP analog is
 * {@link useChatInputModelState}. Mirrors the former SwitchProfileButton's
 * resolution priority so a switch is reflected instantly.
 */
export function useChatInputLlmProfileState(): ChatInputLlmProfileState {
  const { conversationId } = useOptionalConversationId();
  const { data: conversation } = useActiveConversation();
  const { data, isLoading } = useLlmProfiles();
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

  const profiles = data?.profiles ?? [];
  const conversationModel = conversation?.llm_model ?? null;

  // Resolution priority (mirrors the former SwitchProfileButton):
  //   1. Optimistic (just-clicked) — instant feedback before the refetch.
  //   2. Profile stamped on the conversation, validated against the live list
  //      so a since-deleted/renamed profile falls through.
  //   3. Profile whose model matches the running llm_model.
  //   4. User-level active_profile (fallback before the first message).
  const stampedProfile = conversation?.active_profile ?? null;
  const conversationProfile =
    stampedProfile && profiles.some((p) => p.name === stampedProfile)
      ? stampedProfile
      : null;
  const currentProfileName =
    optimisticActiveProfile ??
    conversationProfile ??
    (conversationModel
      ? (profiles.find((p) => p.model === conversationModel)?.name ?? null)
      : (data?.active_profile ?? null));
  const currentProfileModel =
    profiles.find((p) => p.name === currentProfileName)?.model ??
    conversationModel ??
    null;

  // A home pick activates the profile account-wide — on cloud that's the
  // org-gated activate endpoint a member can only 403 on. A cloud `task-…`
  // route has no real conversation id yet, so the per-conversation
  // /switch_profile would 404. Either way the pill still names the active
  // profile; only the selectable rows drop out.
  const isTaskRoute = conversationId?.startsWith("task-") ?? false;
  const canSwitchProfile =
    !isTaskRoute && (!!conversationId || canManageOrgProfiles);

  const selectProfile = useCallback(
    (profileName: string) => {
      if (!canSwitchProfile || profileName === currentProfileName) return;
      switchAndLog(conversationId, profileName);
    },
    [canSwitchProfile, conversationId, currentProfileName, switchAndLog],
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
