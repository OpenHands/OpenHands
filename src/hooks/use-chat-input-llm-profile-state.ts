import { useCallback } from "react";
import { useIsMutating } from "@tanstack/react-query";
import type { ProfileInfo } from "@openhands/typescript-client";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
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
   * Live-switch the running conversation's LLM profile via `/switch_profile`.
   * This surface is only mounted inside a conversation, so a switch always
   * targets that conversation (no home-page activate path here).
   */
  selectProfile: (profileName: string) => void;
}

/**
 * Backs the in-conversation OpenHands LLM-profile switcher (the ACP analog is
 * {@link useChatInputModelState}). Resolves which profile the conversation is
 * running and live-swaps it, mirroring the former SwitchProfileButton's
 * resolution priority so a switch is reflected instantly.
 */
export function useChatInputLlmProfileState(): ChatInputLlmProfileState {
  const { conversationId } = useOptionalConversationId();
  const { data: conversation } = useActiveConversation();
  const { data, isLoading } = useLlmProfiles();
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

  const selectProfile = useCallback(
    (profileName: string) => {
      if (profileName === currentProfileName) return;
      switchAndLog(conversationId, profileName);
    },
    [conversationId, currentProfileName, switchAndLog],
  );

  return {
    profiles,
    currentProfileName,
    currentProfileModel,
    isLoading,
    isSwitching,
    selectProfile,
  };
}
