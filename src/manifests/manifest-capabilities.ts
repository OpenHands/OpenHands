/**
 * Stage 2 — decide whether a deployment can run what a manifest requires.
 *
 * The manifest states what it needs — the features under `requires.features`,
 * and, implicitly, the trigger kinds its form declares. The deployment states
 * what it supports. Neither side is interpreted here beyond set membership, so
 * this stays neutral about what any particular capability means.
 */

import type {
  DeploymentCapabilities,
  SetupActionKind,
  SetupEntry,
} from "./types";

/**
 * "unknown" is a real outcome, not an error: a deployment that cannot be asked
 * must not be treated as one that answered no.
 */
export type SetupCapabilitySupport = boolean | "unknown";

export interface SetupCapabilityAssessment {
  supported: boolean;
  /**
   * Requirement names the deployment did not report, so a block can say which
   * ones rather than only that there were some. Empty when a deployment that
   * reports it is not accepting work blocks the entry, because then no single
   * requirement is the reason.
   */
  unmet: string[];
}

function findUnreported(
  reported: readonly string[],
  required: readonly string[],
): string[] {
  return required.filter((entry) => !reported.includes(entry));
}

function actionSupportsCapabilities(
  features: readonly string[],
  reported: DeploymentCapabilities,
): boolean {
  return findUnreported(reported.features ?? [], features).length === 0;
}

export function supportedActionKinds(
  entry: SetupEntry,
  reported: DeploymentCapabilities,
): SetupActionKind[] {
  const actions = entry.setup.actions ?? {};
  return (Object.keys(actions) as SetupActionKind[]).filter((kind) =>
    actionSupportsCapabilities(actions[kind]?.features ?? [], reported),
  );
}

export function assessCapabilityRequirements(
  entry: SetupEntry,
  reported: DeploymentCapabilities,
): SetupCapabilityAssessment {
  // A deployment that is not accepting work blocks every entry, including one
  // that requires nothing of it.
  if (!reported.ready) return { supported: false, unmet: [] };

  const unmetFeatures = findUnreported(
    reported.features ?? [],
    entry.requires.features ?? [],
  );
  const triggerKinds = Object.keys(entry.setup.form.triggers ?? {});
  const supportsAnyTrigger =
    triggerKinds.length === 0 ||
    triggerKinds.some((kind) => (reported.triggerKinds ?? []).includes(kind));

  const actions = entry.setup.actions ?? {};
  const actionFeatureSets = Object.values(actions).map(
    (action) => action?.features ?? [],
  );
  const supportsAnyAction =
    actionFeatureSets.length === 0 ||
    actionFeatureSets.some((features) =>
      actionSupportsCapabilities(features, reported),
    );
  const unmetActionFeatures = supportsAnyAction
    ? []
    : Array.from(new Set(actionFeatureSets.flat())).filter(
        (feature) => !(reported.features ?? []).includes(feature),
      );

  const unmet = [
    ...unmetFeatures,
    ...unmetActionFeatures,
    ...(supportsAnyTrigger ? [] : triggerKinds),
  ];

  return { supported: unmet.length === 0, unmet };
}
