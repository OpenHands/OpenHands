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

export function assessCapabilityRequirements(
  entry: SetupEntry,
  reported: DeploymentCapabilities,
): SetupCapabilityAssessment {
  // A deployment that is not accepting work blocks every entry, including one
  // that requires nothing of it.
  if (!reported.ready) return { supported: false, unmet: [] };

  const unmet = [
    ...findUnreported(reported.features ?? [], entry.requires.features ?? []),
    ...findUnreported(
      reported.triggerKinds ?? [],
      Object.keys(entry.setup.form.triggers ?? {}),
    ),
  ];

  return { supported: unmet.length === 0, unmet };
}
