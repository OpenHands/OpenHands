import { useMemo } from 'react';
import { useModelStore } from '#/stores/model-store';
import type { LlmProfileSummary } from '#/api/settings-service/profiles-service.api';

/**
 * Hook to get the current model's vision capability.
 *
 * Returns the `supports_vision` flag from the active LLM profile,
 * which tells us whether the current model supports multimodal input.
 *
 * Falls back to `true` if the active profile cannot be determined,
 * preserving existing behavior for users without profile data.
 */
export function useModelCapabilities(conversationId?: string | null) {
  const activeProfileByConversation = useModelStore(
    (s) => s.activeProfileByConversation,
  );
  const entriesByConversation = useModelStore(
    (s) => s.entriesByConversation,
  );

  const supportsVision = useMemo(() => {
    if (!conversationId) {
      return true; // Default to true when no conversation
    }

    // Get the active profile name for this conversation
    const activeProfileName = activeProfileByConversation[conversationId];
    if (!activeProfileName) {
      return true; // Default to true when no active profile
    }

    // Find the profile in the entries to get its supports_vision flag
    const entries = entriesByConversation[conversationId] ?? [];
    for (const entry of entries) {
      if (entry.switchedTo === activeProfileName) {
        // This entry represents a switch to this profile
        // We need to find the profile definition with supports_vision
        // Look through all entries to find the profile with this name
        for (const e of entries) {
          const profile = e.profiles.find((p) => p.name === activeProfileName);
          if (profile) {
            return profile.supports_vision;
          }
        }
      }
      // Also check profiles directly in this entry
      const profile = entry.profiles.find((p) => p.name === activeProfileName);
      if (profile) {
        return profile.supports_vision;
      }
    }

    // Default to true if we can't determine
    return true;
  }, [conversationId, activeProfileByConversation, entriesByConversation]);

  return { supportsVision };
}

/**
 * Get the supports_vision flag from a profile summary.
 * Utility function for cases where we have direct profile access.
 */
export function getProfileSupportsVision(profile: LlmProfileSummary): boolean {
  return profile.supports_vision ?? true;
}
