/**
 * Registry of admitted setup manifests.
 *
 * Stage 1 of the setup flow: load the catalog, decide which entries this host
 * will act on, and index the survivors by id. An entry that fails admission is
 * dropped entirely rather than rendered partially, because everything
 * downstream treats its content as instructions.
 */

import { hasSetupBlock, validateSetupEntry } from "./manifest-validation";
import type { SetupEntry } from "./types";

export interface SetupRegistry {
  /** Entries this host has admitted, in source order. */
  readonly entries: readonly SetupEntry[];
  findById(id: string): SetupEntry | null;
}

export function createSetupRegistry(
  candidates: readonly unknown[],
): SetupRegistry {
  const entries: SetupEntry[] = [];
  const byId = new Map<string, SetupEntry>();

  candidates.forEach((candidate) => {
    // A catalog entry without a setup block is a card, not a setup experience.
    // That is the normal case, so it is skipped rather than reported.
    if (!hasSetupBlock(candidate)) return;

    const { valid, errors } = validateSetupEntry(candidate);
    if (!valid) {
      console.warn("Rejected a setup manifest:", errors.join("; "));
      return;
    }

    const entry = candidate as SetupEntry;
    if (byId.has(entry.id)) {
      console.warn(
        `Rejected a setup manifest: id "${entry.id}" is already registered`,
      );
      return;
    }

    entries.push(entry);
    byId.set(entry.id, entry);
  });

  return {
    entries,
    findById: (id) => byId.get(id) ?? null,
  };
}
