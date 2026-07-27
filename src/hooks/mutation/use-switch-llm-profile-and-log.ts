import { useCallback } from "react";
import { useSwitchLlmProfile } from "#/hooks/mutation/use-switch-llm-profile";

/**
 * Positional wrapper around {@link useSwitchLlmProfile}. The switch's inline
 * "Switched to" message, #1082 metadata persist, and error reporting all live
 * in the mutation itself, so they survive the switcher menu closing on select.
 */
export function useSwitchLlmProfileAndLog() {
  const { mutate, isPending } = useSwitchLlmProfile();

  const switchAndLog = useCallback(
    (conversationId: string | null, profileName: string) => {
      mutate({ conversationId, profileName });
    },
    [mutate],
  );

  return { switchAndLog, isPending };
}
