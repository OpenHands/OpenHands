/**
 * Stage 2 — decide whether a deployment can run what a manifest requires.
 *
 * The manifest states what it needs; the deployment states what it supports.
 * Neither side is interpreted here beyond set membership, so this stays neutral
 * about what any particular capability means.
 */

import type { ManifestCapabilityRequirements } from "./types";

/**
 * "unknown" is a real outcome, not an error: a deployment that cannot be asked
 * must not be treated as one that answered no.
 */
export type ManifestCapabilitySupport = boolean | "unknown";

function supportsAll(
  reported: unknown,
  required: readonly string[] | undefined,
): boolean {
  if (!required || required.length === 0) return true;
  if (!Array.isArray(reported)) return false;
  return required.every((entry) => reported.includes(entry));
}

export function evaluateCapabilityRequirements(
  requires: ManifestCapabilityRequirements,
  reported: Record<string, unknown>,
): boolean {
  if (requires.ready === true && reported.ready !== true) return false;

  return (
    supportsAll(reported.triggerKinds, requires.triggerKinds) &&
    supportsAll(reported.eventSources, requires.eventSources) &&
    supportsAll(reported.eventTypes, requires.eventTypes) &&
    supportsAll(reported.features, requires.features)
  );
}
