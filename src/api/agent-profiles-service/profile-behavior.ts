import type { OpenHandsAgentProfile } from "@openhands/typescript-client";

/** OpenHands profile fields the agent-server stores that the ts-client types omit. */
export interface OpenHandsProfileBehavior {
  /** Deny-list over every skill the profile would load. */
  disabled_skills?: string[];
  /** Allow-list over the public catalog; `null` loads all of it. */
  public_skills?: string[] | null;
  meta_profile_ref?: string | null;
  enable_classify_and_switch_llm_tool?: boolean;
}

export type StoredOpenHandsProfile = OpenHandsAgentProfile &
  OpenHandsProfileBehavior;

/**
 * Whether launching the profile attaches the Model Router's routing tool. On
 * without a name, the server picks the alphabetically first router.
 */
export function profileHasRouter(profile: OpenHandsAgentProfile): boolean {
  return (
    (profile as StoredOpenHandsProfile).enable_classify_and_switch_llm_tool ===
    true
  );
}
