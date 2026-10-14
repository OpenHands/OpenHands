import { useCallback } from "react";
import AgentProfilesService, {
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME,
  type AgentProfileSaveInput,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import { mergeAgentProfileSaveInput } from "#/components/features/settings/agent-profiles/merge-agent-profile-save-input";
import { useSaveAgentProfile } from "#/hooks/mutation/use-save-agent-profile";
import { useActivateAgentProfile } from "#/hooks/mutation/use-activate-agent-profile";

/**
 * The well-known agent profile onboarding configures. Conversations launch from
 * the active AGENT profile (#1571), so onboarding must land the user's choice on
 * an active profile — not just global agent_settings / a standalone LLM profile,
 * which the active profile wouldn't reference. Reusing the seeded "default" name
 * upserts that one profile (its id is preserved on overwrite) rather than
 * spawning a parallel one.
 */
export const ONBOARDING_AGENT_PROFILE_NAME =
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME;

/**
 * Configure and activate the onboarding agent profile from the user's choices:
 * an OpenHands profile pointing at the LLM profile onboarding just created, or
 * an ACP profile for the chosen provider (which owns its own LLM — no key).
 *
 * Best-effort: a failure is swallowed so onboarding is never blocked; the
 * error shows when the user starts a conversation.
 */
export function useApplyOnboardingAgentProfile() {
  const saveProfile = useSaveAgentProfile();
  const activateProfile = useActivateAgentProfile();

  return useCallback(
    async (profile: AgentProfileSaveInput) => {
      try {
        // An OpenHands choice merges over the stored `default`: it carries the
        // user's migrated skills, condenser and verification, which
        // re-onboarding must keep. An ACP choice replaces the profile, so a
        // previous provider's command never runs under the new one.
        const stored =
          profile.agent_kind === "openhands"
            ? await AgentProfilesService.getProfile(
                ONBOARDING_AGENT_PROFILE_NAME,
              )
                .then((detail) => detail.profile)
                .catch(() => null)
            : null;
        await saveProfile.mutateAsync({
          name: ONBOARDING_AGENT_PROFILE_NAME,
          profile: mergeAgentProfileSaveInput(stored, profile),
        });
        // activate needs the stable id; the save response only echoes the name.
        const detail = await AgentProfilesService.getProfile(
          ONBOARDING_AGENT_PROFILE_NAME,
        );
        const id = detail.profile.id;
        if (id) await activateProfile.mutateAsync(id);
      } catch (error) {
        console.error("Failed to configure onboarding agent profile:", error);
      }
    },
    [saveProfile, activateProfile],
  );
}
