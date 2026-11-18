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

const LOCKED_FEATURE_POLICY: FeaturePolicy = {
  version: 1,
  disabled: FEATURE_IDS,
};

function isFeaturePolicy(value: unknown): value is FeaturePolicy {
  if (!value || typeof value !== "object") return false;
  const { version, disabled } = value as Record<string, unknown>;
  return (
    version === 1 &&
    Array.isArray(disabled) &&
    disabled.every((id) => typeof id === "string")
  );
}

/**
 * The operator's policy as injected by static-server, or resolved by the Vite
 * dev server. No policy here means none was configured: every feature is
 * available. Both sources are validated at startup, but a host page can set
 * the window global itself, so a malformed one turns every feature off rather
 * than silently serving the full UI.
 */
export function readInjectedFeaturePolicy(): FeaturePolicy {
  let injected: unknown = null;
  if (typeof window !== "undefined") {
    injected = (window as unknown as Record<string, unknown>)[
      FEATURE_POLICY_WINDOW_KEY
    ];
  }
  if (!injected) injected = __AGENT_CANVAS_DEV_FEATURE_POLICY__;
  if (!injected) return FULL_FEATURE_POLICY;
  return isFeaturePolicy(injected) ? injected : LOCKED_FEATURE_POLICY;
}
