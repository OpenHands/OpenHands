// Whether `llm preset` must also move the seeded `default` agent profile onto
// the LLM profile it activates.
//
// On a local backend, onboarding points `default` at the LLM profile it
// creates (useApplyOnboardingAgentProfile). Activating another LLM profile
// leaves that reference behind. The Agents page then still shows the
// onboarding model on `default`, a launch that picks `default` explicitly
// uses it, and deleting the onboarding profile answers 409 ("referenced by
// 1 agent profile(s)").
//
// Only `default` moves, and only when it is an OpenHands profile whose
// reference names another LLM profile that exists. Named agent profiles are
// deliberate picks. `llm set` never moves it: recipes activate throwaway
// profiles with it and delete them afterwards. A reference to a missing
// profile is the fresh-run seed: the app already falls back to the active LLM
// profile for it, and F13.stale-llm-ref drives that state.

export const DEFAULT_AGENT_PROFILE = "default";

/**
 * @param {object | undefined} agentProfile `profile` from
 *   `GET /api/agent-profiles/default`
 * @param {string[]} llmProfileNames names from `GET /api/profiles`
 * @param {string} target the LLM profile just activated
 * @returns {{ from: string, body: object } | null} the save body for
 *   `POST /api/agent-profiles/default`, or null when nothing should move
 */
export function agentProfileRepoint(agentProfile, llmProfileNames, target) {
  if (agentProfile?.agent_kind !== "openhands") return null;
  const from = agentProfile.llm_profile_ref;
  if (!from || from === target || !llmProfileNames.includes(from)) return null;
  // The save is a whole-profile overwrite: keep every stored field and drop
  // only the identity the server owns (as the Agents editor does).
  const { id, name, revision, ...stored } = agentProfile;
  return { from, body: { ...stored, llm_profile_ref: target } };
}
