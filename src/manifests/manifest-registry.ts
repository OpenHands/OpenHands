/**
 * Registry of admitted extension manifests.
 *
 * Stage 1 of the setup flow: load manifests, decide which ones this host will
 * act on, and index the survivors by the route they mount at and by their id.
 * A manifest that fails admission is dropped entirely rather than rendered
 * partially, because everything downstream treats its content as instructions.
 */

import { validateManifest } from "./manifest-validation";
import type { ExtensionManifest } from "./types";

export interface ManifestRegistry {
  /** Manifests this host has admitted, in source order. */
  readonly manifests: readonly ExtensionManifest[];
  findByRoutePath(pathname: string): ExtensionManifest | null;
  findById(id: string): ExtensionManifest | null;
}

/** Trailing slashes are a URL detail, not a routing difference. */
function normalizeRoutePath(pathname: string): string {
  if (pathname.length > 1 && pathname.endsWith("/")) {
    return pathname.slice(0, -1);
  }
  return pathname;
}

export function createManifestRegistry(
  candidates: readonly unknown[],
): ManifestRegistry {
  const manifests: ExtensionManifest[] = [];
  const byRoutePath = new Map<string, ExtensionManifest>();
  const byId = new Map<string, ExtensionManifest>();

  candidates.forEach((candidate) => {
    const { valid, errors } = validateManifest(candidate);
    if (!valid) {
      console.warn("Rejected an extension manifest:", errors.join("; "));
      return;
    }

    const manifest = candidate as ExtensionManifest;
    if (byId.has(manifest.id)) {
      console.warn(
        `Rejected an extension manifest: id "${manifest.id}" is already registered`,
      );
      return;
    }

    const conflictingRoute = manifest.routes.find((route) =>
      byRoutePath.has(normalizeRoutePath(route.path)),
    );
    if (conflictingRoute) {
      console.warn(
        `Rejected extension manifest "${manifest.id}": route "${conflictingRoute.path}" is already registered`,
      );
      return;
    }

    manifests.push(manifest);
    byId.set(manifest.id, manifest);
    manifest.routes.forEach((route) => {
      byRoutePath.set(normalizeRoutePath(route.path), manifest);
    });
  });

  return {
    manifests,
    findByRoutePath: (pathname) =>
      byRoutePath.get(normalizeRoutePath(pathname)) ?? null,
    findById: (id) => byId.get(id) ?? null,
  };
}
