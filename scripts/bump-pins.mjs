#!/usr/bin/env node
/**
 * Write the upstream release pins into the files that carry them.
 *
 * Run by .github/workflows/bump-pins.yml. That workflow resolves the latest
 * published releases, passes them in as PIN_SDK, PIN_AUTOMATION and
 * PIN_EXTENSIONS, commits whatever this script changed, and keeps one pull
 * request open for it. What gets written:
 *
 * - config/defaults.json: versions.agentServer and versions.automation
 * - package.json: @openhands/typescript-client (always the agent-server
 *   version, see check-sdk-version-sync.mjs) and @openhands/extensions
 * - package-lock.json: the two packages' entries, from the npm registry
 * - the documented agent-server version examples, which
 *   __tests__/scripts/docs-version-sync.test.ts and dev-safe.test.ts assert
 *
 * The lockfile is edited, not regenerated: a fresh `npm install` resolution
 * would re-resolve every other dependency too. That is only safe while the new
 * release keeps the dependencies the lockfile already records for it, so the
 * script stops when they differ and leaves that bump to a person.
 *
 * Idempotent: with the pins already in place it changes nothing.
 *
 * Usage:
 *   PIN_SDK=1.2.3 PIN_AUTOMATION=4.5.6 PIN_EXTENSIONS=7.8.9 node scripts/bump-pins.mjs
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, "..");

const VERSION = /^[0-9]+\.[0-9]+\.[0-9]+$/;
const CLIENT_PACKAGE = "@openhands/typescript-client";
const EXTENSIONS_PACKAGE = "@openhands/extensions";
const DEPENDENCY_FIELDS = [
  "dependencies",
  "optionalDependencies",
  "peerDependencies",
  "peerDependenciesMeta",
];
const escapeRegExp = (text) => text.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");

/** Files that quote versions.agentServer as an example or a test expectation. */
export const EXAMPLE_FILES = [
  ".agents/skills/local-stack-runtime/references/guide.md",
  "scripts/dev-safe.mjs",
  "scripts/check-sdk-version-sync.mjs",
  "__tests__/scripts/dev-safe.test.ts",
];

/**
 * Set the one `"<key>": "X.Y.Z"` pin in a JSON text, keeping its formatting.
 * Only a version-valued entry counts, so `"agentServer": "1.2.3"` under
 * `versions` matches and `"agentServer": "ghcr.io/..."` under `images` does not.
 *
 * @param {string} text
 * @param {string} key
 * @param {string} version
 * @returns {string}
 */
export function setPin(text, key, version) {
  const pin = new RegExp(
    `("${escapeRegExp(key)}"\\s*:\\s*")[0-9]+\\.[0-9]+\\.[0-9]+(")`,
    "g",
  );
  const found = text.match(pin)?.length ?? 0;
  if (found !== 1) {
    throw new Error(`expected one "${key}": "X.Y.Z" pin, found ${found}`);
  }
  return text.replace(pin, `$1${version}$2`);
}

/**
 * Replace a version wherever it stands alone, so bumping 1.2.3 leaves 11.2.3
 * and 1.2.30 untouched.
 *
 * @param {string} text
 * @param {string} from
 * @param {string} to
 * @returns {string}
 */
export function replaceVersion(text, from, to) {
  return text.replace(
    new RegExp(`(?<![0-9.])${escapeRegExp(from)}(?![0-9])`, "g"),
    to,
  );
}

const sortedEntries = (map) =>
  JSON.stringify(
    Object.entries(map ?? {}).sort(([a], [b]) => (a < b ? -1 : 1)),
  );

/**
 * Point one direct dependency's lockfile entry at a release, from that
 * release's registry manifest.
 *
 * @param {string} lockText package-lock.json (lockfileVersion 3)
 * @param {string} name
 * @param {{ version: string, dist: { tarball: string, integrity: string } }} manifest
 * @returns {string}
 */
export function setLockedPackage(lockText, name, manifest) {
  const lock = JSON.parse(lockText);
  const root = lock.packages?.[""]?.dependencies;
  const entry = lock.packages?.[`node_modules/${name}`];
  if (!root?.[name] || !entry) {
    throw new Error(`${name} is not a locked direct dependency`);
  }
  for (const field of DEPENDENCY_FIELDS) {
    if (sortedEntries(entry[field]) !== sortedEntries(manifest[field])) {
      throw new Error(
        `${name}@${manifest.version} changes its ${field}, so package-lock.json ` +
          "needs a real resolution: bump this pin by hand with the pinned npm.",
      );
    }
  }
  root[name] = manifest.version;
  entry.version = manifest.version;
  entry.resolved = manifest.dist.tarball;
  entry.integrity = manifest.dist.integrity;
  return `${JSON.stringify(lock, null, 2)}\n`;
}

/**
 * @param {string} name
 * @param {string} version
 * @returns {Promise<object>} the release's manifest from the npm registry
 */
export async function fetchManifest(name, version) {
  const response = await fetch(`https://registry.npmjs.org/${name}/${version}`);
  if (!response.ok) {
    throw new Error(
      `${name}@${version} is not on npm (HTTP ${response.status})`,
    );
  }
  return response.json();
}

/**
 * @param {string} root repository root
 * @param {{ sdk: string, automation: string, extensions: string }} pins
 * @param {(name: string, version: string) => Promise<object>} getManifest
 * @returns {Promise<string[]>} repository-relative paths that changed
 */
export async function applyPins(root, pins, getManifest = fetchManifest) {
  const read = (rel) => readFileSync(join(root, rel), "utf-8");
  const next = new Map();

  const defaults = read("config/defaults.json");
  const current = JSON.parse(defaults).versions.agentServer;
  next.set(
    "config/defaults.json",
    setPin(
      setPin(defaults, "agentServer", pins.sdk),
      "automation",
      pins.automation,
    ),
  );
  next.set(
    "package.json",
    setPin(
      setPin(read("package.json"), CLIENT_PACKAGE, pins.sdk),
      EXTENSIONS_PACKAGE,
      pins.extensions,
    ),
  );
  if (current !== pins.sdk) {
    for (const rel of EXAMPLE_FILES) {
      next.set(rel, replaceVersion(read(rel), current, pins.sdk));
    }
  }

  let lock = read("package-lock.json");
  for (const [name, version] of [
    [CLIENT_PACKAGE, pins.sdk],
    [EXTENSIONS_PACKAGE, pins.extensions],
  ]) {
    const locked = JSON.parse(lock).packages?.[`node_modules/${name}`];
    if (locked?.version !== version) {
      lock = setLockedPackage(lock, name, await getManifest(name, version));
    }
  }
  next.set("package-lock.json", lock);

  // Written last, so a pin this script refuses leaves the checkout untouched.
  const changed = [...next].filter(([rel, text]) => text !== read(rel));
  for (const [rel, text] of changed) writeFileSync(join(root, rel), text);
  return changed.map(([rel]) => rel);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const pins = {
    sdk: process.env.PIN_SDK,
    automation: process.env.PIN_AUTOMATION,
    extensions: process.env.PIN_EXTENSIONS,
  };
  for (const [name, version] of Object.entries(pins)) {
    if (!VERSION.test(version ?? "")) {
      console.error(
        `PIN_${name.toUpperCase()} must be X.Y.Z, got ${JSON.stringify(version)}`,
      );
      process.exit(1);
    }
  }

  try {
    const changed = await applyPins(projectRoot, pins);
    console.log(
      changed.length
        ? `Updated: ${changed.join(", ")}`
        : "The pins are already in place.",
    );
  } catch (error) {
    console.error(`bump-pins: ${error.message}`);
    process.exit(1);
  }
}
