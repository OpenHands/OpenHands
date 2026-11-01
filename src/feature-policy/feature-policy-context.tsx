import React from "react";
import {
  type FeatureId,
  type FeaturePolicy,
  readInjectedFeaturePolicy,
} from "./feature-policy";

export interface FeaturePolicyChecker {
  isEnabled: (id: FeatureId) => boolean;
}

function createChecker(policy: FeaturePolicy): FeaturePolicyChecker {
  const disabled = new Set(policy.disabled);
  return { isEnabled: (id) => !disabled.has(id) };
}

// Standalone Canvas needs no provider: the default is the injected policy.
const FeaturePolicyContext = React.createContext<FeaturePolicyChecker>(
  createChecker(readInjectedFeaturePolicy()),
);

export function FeaturePolicyProvider({
  policy,
  children,
}: {
  policy: FeaturePolicy;
  children: React.ReactNode;
}) {
  const checker = React.useMemo(() => createChecker(policy), [policy]);
  return (
    <FeaturePolicyContext.Provider value={checker}>
      {children}
    </FeaturePolicyContext.Provider>
  );
}

/**
 * Which features the operator left on. Pass IDs as string literals so a
 * search for an ID finds every place it is gated:
 *
 *   const featurePolicy = useFeaturePolicy();
 *   if (!featurePolicy.isEnabled("backends.add")) …
 */
export function useFeaturePolicy(): FeaturePolicyChecker {
  return React.useContext(FeaturePolicyContext);
}
