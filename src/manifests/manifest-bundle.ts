/**
 * Packing what a bundle entry ships.
 *
 * A bundle's manifest names its files by the repository path they live at; the
 * contents travel in the published `@openhands/extensions` package, because
 * this host has the package and not the repository. That indirection is the
 * whole reason this module exists: everything else about a bundle is derived
 * the same way a prompt entry's request is.
 */

import * as automations from "@openhands/extensions/automations";
import { packTarGzip, type TarFile } from "#/utils/tar-gzip";
import { buildCreatePayload } from "./automation-setup";
import type { SetupEntry, SetupFormValues } from "./types";

/** The rendered configuration, packed beside the entrypoint. */
export const BUNDLE_CONFIG_FILENAME = "config.json";

type BundleFileReader = (id: string) => Record<string, string> | undefined;

/**
 * The files the pinned package ships for this entry.
 *
 * Read defensively: a package predating bundles exports no such function, and
 * an entry that declares a bundle there would otherwise pack an empty archive
 * that fails only once a run tries to execute it.
 */
export function getBundleFiles(id: string): Record<string, string> {
  const read = (automations as { getAutomationBundleFiles?: BundleFileReader })
    .getAutomationBundleFiles;
  const files = read?.(id);
  if (!files || Object.keys(files).length === 0) {
    throw new Error(
      `The published extensions package ships no bundle files for '${id}'.`,
    );
  }
  return files;
}

/**
 * The archive for this entry, with the form's answers rendered into
 * `config.json`.
 *
 * The config is taken from the create payload rather than rendered again here,
 * so what the tarball carries and what the create request records as template
 * provenance cannot disagree.
 */
export async function packBundle(
  entry: SetupEntry,
  values: SetupFormValues,
): Promise<Uint8Array> {
  const bundle = entry.setup.bundle;
  if (!bundle) throw new Error(`'${entry.id}' declares no bundle.`);

  const contents = getBundleFiles(entry.id);
  const missing = Object.keys(bundle.files).filter(
    (name) => typeof contents[name] !== "string",
  );
  if (missing.length > 0) {
    throw new Error(
      `The published extensions package is missing bundle files for ` +
        `'${entry.id}': ${missing.join(", ")}.`,
    );
  }

  const payload = buildCreatePayload(entry, values);
  const template = payload?.template as { config?: unknown } | undefined;

  const files: TarFile[] = Object.keys(bundle.files)
    .sort()
    .map((name) => ({
      name,
      content: contents[name],
      // The service runs the setup script through a shell, so it has to stay
      // executable; nothing else in a bundle is invoked directly.
      mode: name === bundle.setupScript ? 0o755 : 0o644,
    }));

  files.push({
    name: BUNDLE_CONFIG_FILENAME,
    content: `${JSON.stringify(template?.config ?? {}, null, 2)}\n`,
  });

  return packTarGzip(files);
}
