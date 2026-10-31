/**
 * Feature IDs an operator may turn off. Must match `config/features.json`,
 * which the launcher validates a policy against.
 */
export const FEATURE_IDS = ["backends.add"] as const;

export type FeatureId = (typeof FEATURE_IDS)[number];

export interface FeaturePolicy {
  version: 1;
  disabled: readonly FeatureId[];
}

export const FULL_FEATURE_POLICY: FeaturePolicy = { version: 1, disabled: [] };

const FEATURE_POLICY_WINDOW_KEY = "__AGENT_CANVAS_FEATURE_POLICY__";

/**
 * The operator's policy as injected by static-server, or resolved by the Vite
 * dev server. Both sources were validated at startup, so no policy here means
 * none was configured: every feature is available.
 */
export function readInjectedFeaturePolicy(): FeaturePolicy {
  if (typeof window !== "undefined") {
    const injected = (window as unknown as Record<string, unknown>)[
      FEATURE_POLICY_WINDOW_KEY
    ];
    if (injected) return injected as FeaturePolicy;
  }
  if (__AGENT_CANVAS_DEV_FEATURE_POLICY__) {
    return __AGENT_CANVAS_DEV_FEATURE_POLICY__ as FeaturePolicy;
  }
  return FULL_FEATURE_POLICY;
}
