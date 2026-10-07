// The feature map's `Source:` lines, read for drift checks and for mapping a
// set of changed paths to the families that own them.
//
// A `Source:` line lists the main implementation paths of a family in
// backticks. Two shapes carry paths:
//   `src/routes/secrets-settings.tsx`, `src/components/features/settings/secrets-settings/`
//   `src/components/features/automations/` (`automation-card.tsx`, `dashboard/`)
// In the second shape the parenthesized names are relative to the directory
// before them. A backticked token with no `/` and no file extension (a
// function name such as `buildRouterAtStartSystemSuffix`) is a symbol, not a
// path, and is ignored. A trailing `*` in a file name matches by prefix.

const FILE_EXT = /\.(tsx?|m?[jc]s|json|css|md|ya?ml|sh|html|toml|py|txt)$/;

function looksLikePath(token) {
  return token.includes("/") || FILE_EXT.test(token);
}

/**
 * Parse one `Source:` line into path specs.
 * @returns {Array<{ path: string, dir: boolean, glob: boolean }>}
 */
export function parseSourceLine(line) {
  const specs = [];
  const body = line.replace(/^Source:\s*/, "");
  // Tokens: a backticked entry, optionally followed by a parenthesized group
  // of backticked relative entries.
  const entry = /`([^`]+)`(\s*\(([^)]*)\))?/g;
  for (const m of body.matchAll(entry)) {
    const main = m[1].trim();
    if (!looksLikePath(main)) continue;
    specs.push(spec(main));
    if (m[3] && main.endsWith("/")) {
      for (const rel of m[3].matchAll(/`([^`]+)`/g)) {
        const name = rel[1].trim();
        if (!looksLikePath(name)) continue;
        specs.push(spec(main + name));
      }
    }
  }
  return specs;
}

function spec(path) {
  return {
    path: path.replace(/\*$/, "").replace(/\*(?=\.)/, "*"),
    dir: path.endsWith("/"),
    glob: path.includes("*"),
  };
}

/** True when `changed` (a repo-relative file path) belongs to `s`. */
export function specMatches(s, changed) {
  if (s.dir) return changed.startsWith(s.path);
  if (s.glob) {
    const [head, tail = ""] = s.path.split("*");
    return (
      changed.startsWith(head) &&
      changed.endsWith(tail) &&
      !changed.slice(head.length).includes("/")
    );
  }
  return changed === s.path;
}

/** The `Source:` line of a feature file's head (before the first H2). */
export function familyHead(text) {
  const head = text.split(/^## /m)[0];
  const source = head.match(/^Source:.*$/m)?.[0] ?? "";
  return { sources: source ? parseSourceLine(source) : [] };
}

// Paths outside the frontend build and the launcher are not user-facing on
// their own; a change there needs no live recipe (tests, CI, this skill).
const NON_USER_FACING = [
  /^__tests__\//,
  /^__mocks__\//,
  /^\.github\//,
  /^\.agents\//,
  /^\.husky\//,
  /^\.pr\//,
  /^\.screenshots\//,
  /^tests\//,
  /^specs\//,
  /^docs\//,
  // The MSW mock API and dev-only helpers never reach a launched stack.
  /^src\/mocks\//,
  /^src\/dev\//,
  // Build and i18n tooling under scripts/; the launchers there are F26's.
  /^scripts\/[^/]+\.cjs$/,
  /^scripts\/(stryker-diff|generate-icons|brand-dev-electron)\.mjs$/,
  /\.test\.[cm]?[jt]sx?$/,
  /^(README|CHANGELOG|AGENTS|MAINTAINERS)[^/]*\.md$/,
  /^\.[^/]+$/,
];

// Changes here reach every page through the build, so no single family owns
// them; the caller widens to the consumers rather than sampling one screen.
const SHARED_PREFIXES = [
  "src/components/shared/",
  "src/components/ui/",
  "src/ui/",
  "src/styles/",
  "src/index.css",
  "src/tailwind.css",
  "src/themes/",
  "src/i18n/",
  "src/api/",
  "src/hooks/",
  "src/stores/",
  "src/utils/",
  "src/types/",
  "src/context/",
  "src/contexts/",
  "src/services/",
  "src/lib/",
  "src/root.tsx",
  "src/routes.ts",
  "package.json",
  "package-lock.json",
  "vite.config.ts",
  "react-router.config.ts",
  "tailwind.config.js",
  "tsconfig.json",
  "config/",
];

/**
 * Map changed paths to families.
 * @param {Array<{ id: string, file: string, sources: object[] }>} families
 * @param {string[]} changed repo-relative paths
 */
export function affectedFamilies(families, changed) {
  const byFamily = new Map();
  const unmapped = [];
  const shared = [];
  const nonUserFacing = [];
  for (const path of changed) {
    if (NON_USER_FACING.some((re) => re.test(path))) {
      nonUserFacing.push(path);
      continue;
    }
    const owners = families.filter((f) =>
      f.sources.some((s) => specMatches(s, path)),
    );
    if (owners.length) {
      for (const f of owners) {
        if (!byFamily.has(f.id))
          byFamily.set(f.id, { file: f.file, paths: [] });
        byFamily.get(f.id).paths.push(path);
      }
      continue;
    }
    if (SHARED_PREFIXES.some((p) => path.startsWith(p))) shared.push(path);
    else unmapped.push(path);
  }
  return {
    families: [...byFamily.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([id, v]) => ({ id, ...v })),
    shared,
    unmapped,
    nonUserFacing,
  };
}
