/**
 * Resolve the bundled-runtime directory (uv `bin`, Node `node`) for the
 * running Electron process, arch-aware for macOS universal builds.
 *
 * Under a universal Electron binary, `process.arch` reports the slice the
 * OS actually executed — arm64 on Apple Silicon, x64 on Intel hardware or
 * when running under Rosetta (`arch -x86_64`). Universal builds download
 * one runtime tree per arch into `resources/{bin,node}-<arch>`, so the
 * arch-tagged directory is correct by construction: the executing slice
 * can only use runtimes built for that same arch. Single-arch builds keep
 * shipping the legacy flat `resources/{bin,node}` directory, which is the
 * fallback. Returns null when neither candidate exists so callers can emit
 * their own loud missing-runtime warning.
 */
import { existsSync } from "node:fs";
import { join } from "node:path";

/**
 * @param {string} resourcesPath Electron's process.resourcesPath.
 * @param {string} baseName Runtime dir base name ("bin" for uv, "node" for
 *   the bundled Node.js distribution).
 * @param {string} arch CPU arch tag; defaults to the executing process's
 *   arch (overridable so tests can pin arm64/x64).
 * @returns {string | null} The first existing candidate dir, or null when
 *   neither the arch-tagged nor the legacy flat dir exists.
 */
export function resolveBundledRuntimeDir(
  resourcesPath,
  baseName,
  arch = process.arch,
) {
  const candidates = [
    join(resourcesPath, `${baseName}-${arch}`),
    join(resourcesPath, baseName),
  ];
  return candidates.find((dir) => existsSync(dir)) ?? null;
}
