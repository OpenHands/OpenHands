/**
 * The one place that decides which manifests this host is offered.
 *
 * Setup manifests are published by `@openhands/extensions` as an optional
 * `setup` block on a catalog entry, so the catalog is the whole source. Entries
 * without one are skipped, which means the pinned package deciding to ship them
 * is the only step left — there is no wiring to add here.
 *
 * The catalog is passed as `unknown[]` on purpose: admission is a trust
 * boundary, so the host validates the published data rather than trusting the
 * types that shipped beside it.
 */

import * as automationsModule from "@openhands/extensions/automations";
import { AUTOMATION_CATALOG } from "@openhands/extensions/automations";
import { createSetupRegistry, type SetupRegistry } from "./manifest-registry";

export const SETUP_REGISTRY: SetupRegistry = createSetupRegistry(
  AUTOMATION_CATALOG as readonly unknown[],
);

/**
 * The interface manifest the pinned package publishes, if any. It is read off
 * the module namespace so a package that predates the export yields
 * `undefined` rather than a build error; admission in
 * `automation-interface.ts` decides what to do with it.
 */
const automationsExports: object = automationsModule;
export const AUTOMATION_INTERFACE_CANDIDATE: unknown =
  "AUTOMATION_INTERFACE" in automationsExports
    ? automationsExports.AUTOMATION_INTERFACE
    : undefined;
