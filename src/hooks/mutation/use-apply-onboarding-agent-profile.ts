import { useCallback } from "react";
import AgentProfilesService, {
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME,
  type AgentProfileSaveInput,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
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
 * Best-effort: a failure (e.g. an older backend without `/api/agent-profiles`)
 * is swallowed so onboarding is never blocked — the global agent_settings the
 * setup steps already wrote remain the fallback.
 */
export function useApplyOnboardingAgentProfile() {
  const saveProfile = useSaveAgentProfile();
  const activateProfile = useActivateAgentProfile();

  return useCallback(
    async (profile: AgentProfileSaveInput) => {
      try {
        // Overwrite (not merge) is intentional: the `default` profile is the
        // baseline that mirrors the current onboarding choice, not a durable
        // store of per-profile customizations (home-launch resolves it via
        // agent_settings — see `useCreateConversation`). Re-onboarding resets it
        // to the fresh choice; users wanting durable custom config create a
        // named profile, which the editor saves via `mergeAgentProfileSaveInput`.
        await saveProfile.mutateAsync({
          name: ONBOARDING_AGENT_PROFILE_NAME,
          profile,
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
