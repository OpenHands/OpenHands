/**
 * The one place that decides which manifests this host is offered.
 *
 * Manifests are published by extension packages. `@openhands/extensions` does
 * not export any yet, so the only registered source today is the neutrality
 * fixture, which is development-only. Adding the published manifests is a
 * one-line change here and needs no other host change.
 */

import {
  createManifestRegistry,
  type ManifestRegistry,
} from "./manifest-registry";
import { RELEASE_NOTES_DEMO_MANIFEST } from "./fixtures/release-notes-demo-manifest";

function getManifestSources(): unknown[] {
  if (import.meta.env.PROD) return [];
  return [RELEASE_NOTES_DEMO_MANIFEST];
}

export const MANIFEST_REGISTRY: ManifestRegistry =
  createManifestRegistry(getManifestSources());
