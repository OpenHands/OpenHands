import React from "react";

type OnboardingVisibilityOptions = {
  scopeKey: string;
  eligible: boolean;
  dismissed: boolean;
};

/**
 * Keeps an onboarding flow mounted after it opens. Readiness is an entry
 * condition, not an instruction to unmount a flow that just saved its LLM.
 */
export function useOnboardingVisibility({
  scopeKey,
  eligible,
  dismissed,
}: OnboardingVisibilityOptions): boolean {
  const [state, setState] = React.useState(() => ({
    scopeKey,
    isLatched: eligible && !dismissed,
  }));
  const isLatched = state.scopeKey === scopeKey && state.isLatched;

  React.useEffect(() => {
    if (state.scopeKey !== scopeKey) {
      setState({ scopeKey, isLatched: eligible && !dismissed });
      return;
    }
    if (dismissed && state.isLatched) {
      setState({ scopeKey, isLatched: false });
      return;
    }
    if (eligible && !state.isLatched) {
      setState({ scopeKey, isLatched: true });
    }
  }, [dismissed, eligible, scopeKey, state.isLatched, state.scopeKey]);

  return !dismissed && (eligible || isLatched);
}
