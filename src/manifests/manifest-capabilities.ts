/**
 * Stage 2 — decide whether a deployment can run what a manifest requires.
 *
 * The manifest states what it needs — the features under `requires.features`,
 * and, implicitly, the trigger kinds its form declares. The deployment states
 * what it supports. Neither side is interpreted here beyond set membership, so
 * this stays neutral about what any particular capability means.
 */

import type { DeploymentCapabilities, SetupEntry } from "./types";

/**
 * "unknown" is a real outcome, not an error: a deployment that cannot be asked
 * must not be treated as one that answered no.
 */
export type SetupCapabilitySupport = boolean | "unknown";

function supportsAll(
  reported: readonly string[],
  required: readonly string[],
): boolean {
  return required.every((entry) => reported.includes(entry));
}

export function evaluateCapabilityRequirements(
  entry: SetupEntry,
  reported: DeploymentCapabilities,
): boolean {
  if (!reported.ready) return false;

  return (
    supportsAll(reported.features ?? [], entry.requires.features ?? []) &&
    supportsAll(
      reported.triggerKinds ?? [],
      Object.keys(entry.setup.form.triggers ?? {}),
    )
  );
}
