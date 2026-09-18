/**
 * Shared multi-arch resolution for the runtime download scripts
 * (download-node.mjs / download-uv.mjs).
 *
 * macOS universal Electron builds (`npm run build:desktop:universal` with
 * ELECTRON_ARCH=universal) need a per-arch Node.js and uv runtime for BOTH
 * darwin slices — a universal .app that carries only the host-arch runtime
 * silently breaks npx/uvx spawns on the other arch. These helpers decide
 * which arches to download and which resources/<dir> name each one lands in:
 *
 *   single-arch (default): resources/node/ and resources/bin/
 *   universal (macOS only): resources/node-{arm64,x64}/ and resources/bin-{arm64,x64}/
 *
 * The env-var semantics and directory names are pinned together with
 * electron-builder.config.mjs and electron/main.mjs — change one, change all.
 */

/** ELECTRON_ARCH value that selects a macOS universal (dual-arch) download. */
export const UNIVERSAL_ARCH = "universal";

/**
 * The two darwin slices a macOS universal build bundles, in the order the
 * download scripts iterate them. macOS is the only platform Electron
 * supports universal binaries for.
 */
export const DARWIN_UNIVERSAL_ARCHES = ["arm64", "x64"];

/**
 * Resolve the list of runtime architectures to download.
 *
 * - ELECTRON_ARCH unset → `[hostArch]` (today's single-arch behavior,
 *   writes to the legacy resources/{node,bin} directories).
 * - ELECTRON_ARCH=universal + darwin → both slices in
 *   DARWIN_UNIVERSAL_ARCHES order.
 * - ELECTRON_ARCH=universal + non-darwin → throws: there is no universal
 *   build for this platform, so downloading two runtimes would be wasted
 *   work at best and a mis-bundled app at worst.
 * - Any other value (including a concrete arch like "x64") → throws.
 *   Explicit single-arch pinning is deliberately unsupported: a host-arm64
 *   machine downloading an x64-only runtime looks like a typo, and a
 *   silently-wrong cross-arch bundle is much harder to diagnose than a
 *   failed download step.
 */
export function resolveDownloadArches({
  platform = process.platform,
  hostArch = process.arch,
  electronArch = process.env.ELECTRON_ARCH,
} = {}) {
  if (electronArch === undefined) {
    return [hostArch];
  }
  if (electronArch === UNIVERSAL_ARCH) {
    if (platform === "darwin") {
      return [...DARWIN_UNIVERSAL_ARCHES];
    }
    throw new Error(
      `ELECTRON_ARCH=universal is only supported for macOS (darwin) builds; ` +
        `current platform is "${platform}". Remove ELECTRON_ARCH to download ` +
        `the host-arch runtime instead.`,
    );
  }
  throw new Error(
    `Unsupported ELECTRON_ARCH value: "${electronArch}". Leave it unset to ` +
      `download for the host arch (${hostArch}) or set it to ` +
      `"${UNIVERSAL_ARCH}" for a macOS universal build.`,
  );
}

/**
 * Name of the resources subdirectory for a runtime download.
 * `base` is "node" or "bin". Multi-arch (universal) downloads get per-arch
 * suffixed directories so both slices can coexist; single-arch downloads
 * keep the legacy unsuffixed name.
 */
export function resourceDirName(base, arch, multi) {
  return multi ? `${base}-${arch}` : base;
}
