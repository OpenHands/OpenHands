import { useEffect, useRef } from "react";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useActivateLlmProfile } from "#/hooks/mutation/use-activate-llm-profile";
import { isUsableLlmProfile } from "#/utils/usable-llm-profile";

/**
 * Local-mode UX policy: keep an LLM profile active whenever at least one
 * exists, so the agent always has a usable LLM without a manual "activate"
 * step. This lives in the client on purpose — the agent-server is a neutral
 * API that returns `active_profile` as stored (possibly pointing at a deleted
 * profile); other consumers (e.g. SaaS) own the LLM differently.
 *
 * When profiles exist but none is the active one — never activated, or the
 * active profile was deleted — it activates the first usable profile (API key,
 * provider connection, or Ollama URL), falling back to the first profile.
 * If the active profile exists but is not usable, it promotes a usable one.
 */
export function useEnsureActiveProfile(): void {
  const { backend } = useActiveBackend();
  const isLocal = backend.kind === "local";
  const { data: profilesData } = useLlmProfiles();
  const { mutate: activate, isPending } = useActivateLlmProfile();

  // Remember the last profile we tried to activate so we don't re-fire while
  // the mutation + refetch settle, or hammer a profile whose activation fails.
  const attemptedRef = useRef<string | null>(null);

  // A backend switch is a clean slate for the above guard.
  useEffect(() => {
    attemptedRef.current = null;
  }, [backend.id]);

  useEffect(() => {
    if (!isLocal || isPending || !profilesData) return;

    const { profiles, active_profile: activeProfile } = profilesData;
    const active = profiles.find((profile) => profile.name === activeProfile);

    if (isUsableLlmProfile(active)) {
      attemptedRef.current = null;
      return;
    }

    if (profiles.length === 0) {
      attemptedRef.current = null;
      return;
    }

    const target =
      profiles.find((profile) => isUsableLlmProfile(profile)) ??
      (active ? null : profiles[0]);
    if (!target || target.name === active?.name) {
      attemptedRef.current = null;
      return;
    }
    if (attemptedRef.current === target.name) return;
    attemptedRef.current = target.name;
    activate(target.name);
  }, [isLocal, profilesData, isPending, activate]);
}
