import { useCallback, useEffect } from "react";
import { useIsMutating, useQueryClient } from "@tanstack/react-query";
import { WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME } from "#/api/agent-profiles-service/agent-profiles-service.api";
import { mergeStoredConversationMetadata } from "#/api/conversation-metadata-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  llmProfileMirrored,
  updateAgentProfile,
} from "#/hooks/mutation/sync-agent-profiles";
import { useAgentProfiles } from "#/hooks/query/use-agent-profiles";
import { useAgentProfileAvailability } from "#/hooks/use-agent-profile-availability";
import type { ProfileInfo } from "@openhands/typescript-client";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useCanManageOrgProfiles } from "#/hooks/use-can-manage-org-profiles";
import { useSwitchLlmProfileAndLog } from "#/hooks/mutation/use-switch-llm-profile-and-log";
import { SWITCH_LLM_PROFILE_MUTATION_KEY } from "#/hooks/mutation/use-switch-llm-profile";
import { useModelStore } from "#/stores/model-store";

// Tracks the `(backendId, orgId)` for which a home-page LLM profile pick is
// pending before the first message. Read and cleared only by `HomeChatLauncher`
// (so non-home `useCreateConversation` callers neither consume nor are affected
// by it), and keyed by backend + org so a pick on one environment never carries
// over to another.
let pendingHomePickKey: string | null = null;

const toHomePickKey = (
  backendId: string,
  orgId: string | null | undefined,
): string => JSON.stringify([backendId, orgId ?? null]);

export function markHomeModelPick(
  backendId: string,
  orgId?: string | null,
): void {
  pendingHomePickKey = toHomePickKey(backendId, orgId);
}

export function hasPendingHomeModelPick(
  backendId: string,
  orgId?: string | null,
): boolean {
  return pendingHomePickKey === toHomePickKey(backendId, orgId);
}

export function clearPendingHomeModelPick(
  backendId?: string,
  orgId?: string | null,
): void {
  if (
    backendId === undefined ||
    pendingHomePickKey === toHomePickKey(backendId, orgId)
  ) {
    pendingHomePickKey = null;
  }
}

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
  const { data: conversation, isFetching: isConversationFetching } =
    useActiveConversation();
  const { data, isLoading } = useLlmProfiles();
  const queryClient = useQueryClient();
  const { backend, orgId } = useActiveBackend();
  const { data: agentProfiles } = useAgentProfiles({
    enabled: !conversationId,
  });
  const agentProfilesAvailable = useAgentProfileAvailability();
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
  const rawOptimistic = useModelStore((s) =>
    conversationId ? s.activeProfileByConversation[conversationId] : undefined,
  );
  const homeLlmProfile = useModelStore((s) => s.homeLlmProfile);
  const setHomeLlmProfile = useModelStore((s) => s.setHomeLlmProfile);

  const profiles = data?.profiles ?? [];
  const conversationModel = conversation?.llm_model ?? null;
  // The conversation query serves stale cached `llm_model` during the
  // background refetch a successful switch triggers (invalidateQueries in
  // onSuccess). Until that refetch lands, `conversationModel` still reflects
  // the pre-switch model, so treating it as authoritative would (a) drop the
  // just-written optimistic entry on render and (b) make the repair effect
  // rewrite the just-persisted stamp back to the pre-switch profile with a
  // fresh stamped_at — which can then suppress seedModelSwitchesFromHistory's
  // corrective re-stamp on reload. Only treat the model as confirmed when the
  // fetch is not in flight.
  const conversationModelConfirmed =
    conversationModel !== null && !isConversationFetching;
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

  // An in-flight switch's optimistic entry wins during the pending switch window.
  // Outside that window, a stale optimistic entry that conflicts with confirmed
  // conversation data does not override the confirmed resolution.
  const optimisticProfileObj = rawOptimistic
    ? profiles.find((p) => p.name === rawOptimistic)
    : undefined;
  const isOptimisticConflicting =
    !isSwitching &&
    Boolean(
      conversationModelConfirmed &&
      optimisticProfileObj &&
      optimisticProfileObj.model !== conversationModel,
    );
  const optimisticActiveProfile = isOptimisticConflicting
    ? undefined
    : rawOptimistic;

  // Resolution priority (mirrors the former SwitchProfileButton):
  //   1. Optimistic (just-clicked) — during an in-flight switch, or if not
  //      conflicting with confirmed conversation data.
  //   2. Profile stamped on the conversation, validated against the live list
  //      AND confirmed to match the conversation's running llm_model. A stale
  //      browser stamp whose model disagrees with backend llm_model falls through
  //      to model matching.
  //   3. Profile whose model matches the running llm_model.
  //   4. Before the first message, a cloud member's home pick, else the
  //      active agent profile's own LLM, which the launch runs, else the
  //      user-level active_profile.
  const stampedProfile = conversation?.active_profile ?? null;
  const stampedProfileObj = stampedProfile
    ? profiles.find((p) => p.name === stampedProfile)
    : undefined;
  const isStampConflicting = Boolean(
    stampedProfileObj &&
    conversationModelConfirmed &&
    stampedProfileObj.model !== conversationModel,
  );
  const conversationProfile =
    stampedProfileObj && !isStampConflicting ? stampedProfile : null;

  // When a confirmed fetch disagrees with browser metadata, repair or clear the
  // stored stamp without dropping other metadata fields. Skip while the
  // conversation query is refetching (e.g. after a switch's invalidateQueries):
  // the cached llm_model is stale there, so running the repair would rewrite a
  // just-persisted stamp back to the pre-switch profile and clear the fresh
  // optimistic entry. The effect re-runs when the refetch lands.
  useEffect(() => {
    if (!conversationId || !conversationModelConfirmed) return;
    if (stampedProfile) {
      const stamped = profiles.find((p) => p.name === stampedProfile);
      if (stamped && stamped.model !== conversationModel) {
        const repaired =
          profiles.find((p) => p.model === conversationModel)?.name ?? null;
        mergeStoredConversationMetadata(conversationId, {
          active_profile: repaired,
          stamped_at: new Date().toISOString(),
        });
      }
    }
    if (!isSwitching) {
      const opt =
        useModelStore.getState().activeProfileByConversation[conversationId];
      if (opt) {
        const optProfile = profiles.find((p) => p.name === opt);
        if (optProfile && optProfile.model !== conversationModel) {
          useModelStore.getState().clearActiveProfile(conversationId);
        }
      }
    }
  }, [
    conversationId,
    conversationModelConfirmed,
    conversationModel,
    stampedProfile,
    profiles,
    isSwitching,
  ]);

  // An instance with Agent Profiles off launches without the activated
  // profile, so that profile neither names nor takes the home pick.
  const activeAgentProfile = agentProfilesAvailable
    ? agentProfiles?.profiles?.find(
        (profile) =>
          profile.id != null &&
          profile.id === agentProfiles.active_agent_profile_id,
      )
    : undefined;
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
      if (!canSwitchProfile) return;
      const selectedProfile = profiles.find((p) => p.name === profileName);
      const isAlreadyRunningInConversation =
        conversationId &&
        conversationModel &&
        selectedProfile?.model === conversationModel &&
        profileName === currentProfileName;
      if (
        profileName === currentProfileName &&
        (isAlreadyRunningInConversation || !conversationId)
      ) {
        return;
      }
      if (!conversationId) {
        // A home-page LLM profile pick is an explicit model choice made before
        // the first message: mark it for this (backend, org) so the next home
        // launch from `HomeChatLauncher` passes `suppressRouterAtStart: true`
        // for that single conversation without affecting non-home launches or
        // other backends/orgs.
        markHomeModelPick(backend.id, orgId);
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
      backend.id,
      orgId,
      canManageOrgProfiles,
      canSwitchProfile,
      conversationId,
      conversationModel,
      currentProfileName,
      profiles,
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
