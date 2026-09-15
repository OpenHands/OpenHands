/**
 * External skills source resolution for the launcher `--skills` flag.
 *
 * `agent-canvas --skills <source>` (repeatable, or OH_SKILLS_SOURCES) lets a
 * team attach a shared skills checkout at startup without copying SKILL.md
 * files into every user's or project's .agents directory. Sources are
 * resolved once at launch, scanned for valid Agent Skills, and written to a
 * manifest the frontend reads via an injected window global
 * (`__AGENT_CANVAS_EXTERNAL_SKILLS__`).
 *
 * Deliberately filesystem-only: a source is cloned (git) and its SKILL.md
 * files are read as text. No repository scripts, package-manager hooks, or
 * setup code are ever executed — git clone itself runs no repository hooks.
 *
 * Every source is best-effort: a clone failure, unreadable directory, or
 * malformed SKILL.md produces a stderr warning and the remaining sources are
 * still loaded.
 */

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, isAbsolute, join, resolve } from "node:path";

// Agent Skills naming rules, matching the agent-server SDK's strict loader:
// lowercase alphanumerics separated by single hyphens, at most 64 chars, and
// the frontmatter `name` must equal the skill's directory name.
const SKILL_NAME_PATTERN = /^[a-z0-9]+(-[a-z0-9]+)*$/;
const SKILL_NAME_MAX_LENGTH = 64;

// `host/path/...` without a scheme — e.g. github.com/acmecorp/skills or
// gitlab.com/platform/team-skills. A dot in the first segment is what keeps a
// relative path like `acme/skills` on the local-directory path.
const GIT_HOST_SHORTHAND_PATTERN = /^[\w.-]+\.[a-zA-Z]{2,}\/\S+$/;

const GIT_URL_PATTERN = /^([a-z][a-z0-9+.-]*):\/\//i;

const GIT_CLONE_TIMEOUT_MS = 120_000;

/**
 * @typedef {object} ParsedSkill
 * @property {string} name
 * @property {string | null} description
 * @property {string[]} triggers
 * @property {string} content
 * @property {string | null} license
 * @property {string | null} compatibility
 * @property {string | null} version
 * @property {string[]} allowedTools
 * @property {string} path - Absolute SKILL.md location on the launcher host.
 */

/**
 * @typedef {object} ExternalSkillsManifest
 * @property {number} version
 * @property {{id: string, source: string, kind: string, path: string, skills: string[]}[]} sources
 * @property {{name: string, description: string | null, triggers: string[], content: string, license: string | null, compatibility: string | null, version: string | null, allowed_tools: string[] | null, path: string, source: string, source_id: string}[]} skills
 */

/**
 * @typedef {{kind: "local", dir: string} | {kind: "git", url: string} | {kind: "invalid", reason: string}} ClassifiedSource
 */

/**
 * Collect the repeatable `--skills <source>` / `--skills=<source>` flag from a
 * raw argv slice. Used by launchers that have no parseArgs of their own
 * (dev-safe.mjs); launchers with a parser call it once per flag instead.
 *
 * @param {string[]} argv
 * @returns {Array<string | undefined>}
 */
export function collectSkillsSourcesFromArgv(argv) {
  const sources = [];
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--skills") {
      sources.push(argv[++i]);
    } else if (arg.startsWith("--skills=")) {
      sources.push(arg.slice("--skills=".length));
    }
  }
  return sources;
}

/**
 * Split OH_SKILLS_SOURCES — comma- or newline-separated list of sources.
 *
 * @param {unknown} value
 * @returns {string[]}
 */
export function parseSkillsSourcesEnv(value) {
  if (typeof value !== "string" || !value.trim()) return [];
  return value
    .split(/[\n,]/)
    .map((entry) => entry.trim())
    .filter(Boolean);
}

function expandHome(source) {
  if (source === "~") return homedir();
  if (source.startsWith("~/") || source.startsWith("~\\")) {
    return join(homedir(), source.slice(2));
  }
  return source;
}

/**
 * Decide what a `--skills` value points at.
 *
 * Order matters: an existing local path always wins, so a directory that
 * happens to be named `github.com` on disk is not mistaken for a repo. Only
 * then do URL shapes (`https://`, `ssh://`, `git@…:`, `file://`, …) and the
 * `host/path` shorthand get a git interpretation.
 *
 * @param {string} source
 * @param {{cwd?: string}} [options]
 * @returns {ClassifiedSource}
 */
export function classifySkillsSource(source, { cwd = process.cwd() } = {}) {
  const expanded = expandHome(source);
  const resolvedPath = isAbsolute(expanded) ? expanded : resolve(cwd, expanded);

  if (existsSync(resolvedPath)) {
    if (!statSync(resolvedPath).isDirectory()) {
      return {
        kind: "invalid",
        reason: `not a directory: ${resolvedPath}`,
      };
    }
    return { kind: "local", dir: resolvedPath };
  }

  if (GIT_URL_PATTERN.test(source) || source.startsWith("git@")) {
    return { kind: "git", url: source };
  }

  if (GIT_HOST_SHORTHAND_PATTERN.test(source)) {
    return { kind: "git", url: `https://${source}` };
  }

  return {
    kind: "invalid",
    reason:
      `path does not exist and is not a git source: ${source} ` +
      `(expected a local directory, a git URL, or host/path shorthand)`,
  };
}

/**
 * The readable part of a source id: the basename (minus a `.git` suffix)
 * slugified into the Agent Skills alphabet. For a git source this is also the
 * expected name of a repo that is itself one skill — the clone lands in a
 * synthetic `<slug>-<hash>` cache directory, so the SDK's
 * name-matches-directory rule has to compare against this instead.
 */
function sourceSlugFor(canonical) {
  let tail = canonical.replace(/\.git$/, "").replace(/[/\\:]+$/, "");
  tail = basename(tail) || tail;
  return (
    tail
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 32) || "source"
  );
}

/**
 * Stable, collision-safe identity for one source: a readable slug plus a hash
 * of the canonical source string (absolute path for locals, clone URL for
 * git). The same `--skills` value yields the same id across restarts, which
 * is what makes persisted enablement keys stable.
 *
 * @param {string} kind
 * @param {string} canonical
 * @returns {string}
 */
export function sourceIdFor(kind, canonical) {
  const hash = createHash("sha256")
    .update(`${kind}:${canonical}`)
    .digest("hex")
    .slice(0, 8);

  return `${sourceSlugFor(canonical)}-${hash}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// SKILL.md parsing — a deliberately small YAML subset, enough for the Agent
// Skills frontmatter conventions (scalars, `- item` lists, `[a, b]` flow
// lists, and `key:` block scalars which we skip).
// ─────────────────────────────────────────────────────────────────────────────

function unquoteYamlScalar(value) {
  const trimmed = value.trim();
  if (trimmed.length >= 2) {
    const first = trimmed[0];
    const last = trimmed[trimmed.length - 1];
    if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
      return trimmed.slice(1, -1);
    }
  }
  return trimmed;
}

function parseFlowList(value) {
  // `[a, b, "c"]` — no nesting or escaped commas; frontmatter conventions
  // only need the simple case.
  const inner = value.trim().slice(1, -1).trim();
  if (!inner) return [];
  return inner.split(",").map(unquoteYamlScalar).filter(Boolean);
}

function parseFrontmatterYaml(raw) {
  const meta = {};
  const lines = raw.split(/\r?\n/);
  let pendingListKey = null;

  for (const line of lines) {
    if (!line.trim() || line.trimStart().startsWith("#")) continue;

    const indented = /^\s/.test(line);

    // `- item` under a `key:` line collects into a list.
    const listItem = line.match(/^\s+-\s+(.*)$/);
    if (indented && listItem && pendingListKey) {
      if (!Array.isArray(meta[pendingListKey])) meta[pendingListKey] = [];
      meta[pendingListKey].push(unquoteYamlScalar(listItem[1]));
      continue;
    }

    // Any other indented line either belongs to a block scalar (skipped) or
    // is a nested mapping (not part of the conventions we consume).
    if (indented) continue;

    const kv = line.match(/^([A-Za-z0-9_-]+):(?:[ \t]+(.*))?$/);
    if (!kv) continue;
    const [, key, value = ""] = kv;
    const trimmed = value.trim();

    if (trimmed === "") {
      // Either a `- list` follows or a nested mapping does; the list branch
      // above handles the former.
      pendingListKey = key;
      meta[key] = [];
    } else if (trimmed.startsWith("|") || trimmed.startsWith(">")) {
      // Block scalar: the indented body is skipped by the branch above; a
      // description long enough to need one is better left unset than
      // mis-parsed.
      pendingListKey = null;
      meta[key] = "";
    } else if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
      pendingListKey = null;
      meta[key] = parseFlowList(trimmed);
    } else {
      pendingListKey = null;
      meta[key] = unquoteYamlScalar(trimmed);
    }
  }

  return meta;
}

/**
 * Split a SKILL.md into frontmatter and body. A file without a `---` fence
 * yields empty metadata (the SDK then takes the name from the directory); a
 * file whose fence never closes returns null — malformed, not absent.
 */
function splitFrontmatter(text) {
  const normalized = text.replace(/^\uFEFF/, "");
  if (!normalized.startsWith("---")) {
    return { meta: {}, body: normalized };
  }
  const match = normalized.match(
    /^---[ \t]*\r?\n([\s\S]*?)\r?\n---[ \t]*\r?\n?/,
  );
  if (!match) return null;
  return {
    meta: parseFrontmatterYaml(match[1]),
    body: normalized.slice(match[0].length),
  };
}

function asStringList(value) {
  if (Array.isArray(value)) return value.filter((v) => typeof v === "string");
  if (typeof value === "string" && value.trim()) {
    // `triggers: a b` and `allowed-tools: Read Write` appear in the wild as
    // space-separated scalars.
    return value.split(/[\s,]+/).filter(Boolean);
  }
  return [];
}

/**
 * Parse and validate one SKILL.md against the Agent Skills conventions.
 * Returns null (having warned) for anything the SDK's strict loader would
 * reject: a broken frontmatter fence, a name that is not the skill
 * directory's name, or a name outside the allowed pattern.
 *
 * `expectedName` is the name the skill's directory carries: the real
 * directory basename everywhere except the root SKILL.md of a git clone,
 * whose on-disk directory is the synthetic `<slug>-<hash>` cache target —
 * there the repository name stands in for the skill directory.
 *
 * @param {string} filePath
 * @param {string} expectedName
 * @param {(message: string) => void} warn
 * @returns {ParsedSkill | null}
 */
export function parseSkillMarkdown(filePath, expectedName, warn) {
  let raw;
  try {
    raw = readFileSync(filePath, "utf8");
  } catch (error) {
    warn(`cannot read ${filePath}: ${error.message}`);
    return null;
  }

  const parts = splitFrontmatter(raw);
  if (parts === null) {
    warn(`skipping ${filePath}: frontmatter fence never closes`);
    return null;
  }

  const name =
    typeof parts.meta.name === "string" ? parts.meta.name.trim() : expectedName;

  if (parts.meta.name !== undefined && name !== expectedName) {
    warn(
      `skipping ${filePath}: frontmatter name "${name}" does not match ` +
        `skill directory name "${expectedName}"`,
    );
    return null;
  }
  if (!SKILL_NAME_PATTERN.test(name) || name.length > SKILL_NAME_MAX_LENGTH) {
    warn(
      `skipping ${filePath}: "${name}" is not a valid Agent Skills name ` +
        `(lowercase letters, digits and single hyphens, ≤ ${SKILL_NAME_MAX_LENGTH} chars)`,
    );
    return null;
  }

  const description =
    typeof parts.meta.description === "string" && parts.meta.description.trim()
      ? parts.meta.description.trim()
      : null;
  if (!description) {
    warn(
      `${filePath}: no description in frontmatter — the agent cannot ` +
        `decide when to trigger this skill`,
    );
  }

  const content = parts.body.trim();
  if (!content) {
    warn(`skipping ${filePath}: SKILL.md has no instructions body`);
    return null;
  }

  return {
    name,
    description,
    triggers: asStringList(parts.meta.triggers),
    content,
    license: typeof parts.meta.license === "string" ? parts.meta.license : null,
    compatibility:
      typeof parts.meta.compatibility === "string"
        ? parts.meta.compatibility
        : null,
    version: typeof parts.meta.version === "string" ? parts.meta.version : null,
    allowedTools: asStringList(
      parts.meta["allowed-tools"] ?? parts.meta.allowed_tools,
    ),
    path: filePath,
  };
}

// Directories that hold `<name>/SKILL.md` skill layouts inside a source.
// `skills/` is the org-repo convention the agent-server itself uses;
// `.agents/skills/` and `microagents/` cover project- and legacy-style
// checkouts.
const SKILL_CONTAINER_DIRS = [
  "skills",
  join(".agents", "skills"),
  "microagents",
];

// Top-level children that are never skill directories — `.git`, the container
// dirs themselves (already scanned), and vendored dependencies.
const TOP_LEVEL_SKIP_DIRS = new Set([
  "skills",
  ".agents",
  "microagents",
  "node_modules",
]);

function skillDirsUnder(containerDir) {
  let entries;
  try {
    entries = readdirSync(containerDir, { withFileTypes: true });
  } catch {
    return [];
  }
  const dirs = [];
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.startsWith(".")) continue;
    const skillMd = join(containerDir, entry.name, "SKILL.md");
    if (existsSync(skillMd))
      dirs.push({ dir: join(containerDir, entry.name), file: skillMd });
  }
  return dirs;
}

/**
 * Find every `SKILL.md` a source directory exposes:
 *   - `<dir>/SKILL.md` — the directory itself is one skill,
 *   - `<dir>/{skills,.agents/skills,microagents}/<name>/SKILL.md` — the
 *     conventions the agent-server's own loaders use, and
 *   - `<dir>/<name>/SKILL.md` — a bare directory of skills with no wrapper.
 */
function findSkillFiles(rootDir) {
  const found = new Map();

  const rootSkill = join(rootDir, "SKILL.md");
  if (existsSync(rootSkill)) {
    found.set(rootSkill, { dir: rootDir, file: rootSkill });
  }

  for (const sub of SKILL_CONTAINER_DIRS) {
    for (const entry of skillDirsUnder(join(rootDir, sub))) {
      if (!found.has(entry.file)) found.set(entry.file, entry);
    }
  }

  for (const entry of skillDirsUnder(rootDir)) {
    const childName = basename(entry.dir);
    if (TOP_LEVEL_SKIP_DIRS.has(childName)) continue;
    if (!found.has(entry.file)) found.set(entry.file, entry);
  }

  return [...found.values()];
}

// ─────────────────────────────────────────────────────────────────────────────
// Git acquisition — clone only, never anything the repository ships.
// ─────────────────────────────────────────────────────────────────────────────

function gitEnv() {
  return {
    ...process.env,
    // Fail fast instead of hanging on a credential prompt for a private repo
    // the user cannot authenticate.
    GIT_TERMINAL_PROMPT: "0",
    GIT_LFS_SKIP_SMUDGE: "1",
  };
}

function sanitizeGitError(text, url) {
  return String(text ?? "")
    .split(url)
    .join("<url>")
    .trim();
}

/**
 * Clone (or refresh the cached clone of) a git source into `cacheDir`.
 * `--depth 1` keeps this cheap; a previously cloned source is pulled
 * best-effort and otherwise reused as-is — a stale copy beats no copy.
 *
 * @param {string} url
 * @param {string} cacheDir
 * @param {{warn?: (message: string) => void, info?: (message: string) => void}} [options]
 * @returns {string} The checked-out directory.
 */
export function checkoutSkillsRepo(url, cacheDir, { warn, info } = {}) {
  mkdirSync(cacheDir, { recursive: true });
  const target = join(cacheDir, sourceIdFor("git", url));

  if (existsSync(join(target, ".git"))) {
    const pull = spawnSync(
      "git",
      ["-C", target, "pull", "--ff-only", "--quiet"],
      {
        env: gitEnv(),
        timeout: GIT_CLONE_TIMEOUT_MS,
        encoding: "utf8",
      },
    );
    if (pull.status !== 0) {
      warn?.(
        `--skills: could not refresh cached clone of ${url}; ` +
          `reusing previous checkout (${sanitizeGitError(pull.stderr || pull.error?.message, url)})`,
      );
    } else {
      info?.(`--skills: refreshed ${url}`);
    }
    return target;
  }

  const clone = spawnSync(
    "git",
    ["clone", "--depth", "1", "--quiet", url, target],
    { env: gitEnv(), timeout: GIT_CLONE_TIMEOUT_MS, encoding: "utf8" },
  );
  if (clone.status !== 0 || !existsSync(target)) {
    const detail = sanitizeGitError(clone.stderr || clone.error?.message, url);
    throw new Error(
      `git clone failed${detail ? ` — ${detail}` : ""}. ` +
        `Check the URL and that credentials are available non-interactively ` +
        `(e.g. a git credential helper or an ssh agent).`,
    );
  }
  info?.(`--skills: cloned ${url}`);
  return target;
}

// ─────────────────────────────────────────────────────────────────────────────
// Resolution — one manifest for the frontend + conversation payload.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Resolve every `--skills` source into a manifest:
 *   {
 *     version: 1,
 *     skills:  [{ name, description, triggers, content, license,
 *                 compatibility, version, path, source, source_id }],
 *     sources: [{ id, source, kind, path, skills: [name, …] }],
 *   }
 *
 * `path` is the absolute SKILL.md location on the launcher host so the local
 * agent-server can resolve relative resources (scripts/, references/) when a
 * conversation opts the skill in. `source` is the argument as supplied, for
 * display.
 *
 * @param {string[] | undefined} sources
 * @param {{cacheDir?: string, warn?: (message: string) => void, info?: (message: string) => void}} [options]
 * @returns {ExternalSkillsManifest}
 */
export function resolveExternalSkillsSources(
  sources,
  { cacheDir, warn, info } = {},
) {
  /** @type {ExternalSkillsManifest} */
  const manifest = { version: 1, sources: [], skills: [] };
  const warnFn = warn ?? ((message) => console.error(`Warning: ${message}`));
  const infoFn = info ?? (() => {});

  for (const rawSource of sources ?? []) {
    const source = typeof rawSource === "string" ? rawSource.trim() : "";
    if (!source) {
      warnFn("--skills: ignoring empty source value");
      continue;
    }

    let rootDir;
    let kind;
    let canonical;
    try {
      const classified = classifySkillsSource(source);
      if (classified.kind === "invalid") {
        warnFn(`--skills: ${classified.reason}`);
        continue;
      }
      kind = classified.kind;
      if (classified.kind === "local") {
        rootDir = classified.dir;
        canonical = classified.dir;
      } else {
        canonical = classified.url;
        rootDir = checkoutSkillsRepo(classified.url, cacheDir, {
          warn: warnFn,
          info: infoFn,
        });
      }
    } catch (error) {
      warnFn(`--skills: ${source}: ${error.message}`);
      continue;
    }

    const sourceId = sourceIdFor(kind, canonical);
    // A git clone's root directory is the synthetic `<slug>-<hash>` cache
    // target, so a repo that is itself one skill (root SKILL.md) checks its
    // frontmatter name against the repository name instead. Local sources
    // and nested skills keep using the real directory basename.
    const rootExpectedName =
      kind === "git" ? sourceSlugFor(canonical) : basename(rootDir);

    const sourceSkills = [];
    const seenNames = new Set();
    for (const { dir, file } of findSkillFiles(rootDir)) {
      const parsed = parseSkillMarkdown(
        file,
        dir === rootDir ? rootExpectedName : basename(dir),
        (message) => warnFn(`--skills ${source}: ${message}`),
      );
      if (!parsed) continue;
      if (seenNames.has(parsed.name)) {
        warnFn(
          `--skills ${source}: ignoring duplicate skill name "${parsed.name}" ` +
            `(${file})`,
        );
        continue;
      }
      seenNames.add(parsed.name);
      sourceSkills.push(parsed);
      manifest.skills.push({
        name: parsed.name,
        description: parsed.description,
        triggers: parsed.triggers,
        content: parsed.content,
        license: parsed.license,
        compatibility: parsed.compatibility,
        version: parsed.version,
        allowed_tools: parsed.allowedTools.length ? parsed.allowedTools : null,
        path: parsed.path,
        source,
        source_id: sourceId,
      });
    }

    if (sourceSkills.length === 0) {
      warnFn(
        `--skills: ${source}: no valid Agent Skills found ` +
          `(looked for SKILL.md under skills/, .agents/skills/, microagents/, ` +
          `and top-level directories)`,
      );
      continue;
    }

    manifest.sources.push({
      id: sourceId,
      source,
      kind,
      path: rootDir,
      skills: sourceSkills.map((skill) => skill.name),
    });
    infoFn(
      `--skills: loaded ${sourceSkills.length} skill(s) from ${source} ` +
        `(${sourceSkills.map((skill) => skill.name).join(", ")})`,
    );
  }

  return manifest;
}

/**
 * Persist the manifest the frontend will be pointed at.
 *
 * @param {string} filePath
 * @param {ExternalSkillsManifest} manifest
 * @returns {string} The path written.
 */
export function writeExternalSkillsManifest(filePath, manifest) {
  mkdirSync(dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  return filePath;
}
