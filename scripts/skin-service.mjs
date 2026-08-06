/**
 * Skin service — install, run, and sync a "skin" for this Agent Canvas
 * instance.
 *
 * A skin is a git-backed package (GitHub only for v1) that:
 *   • serializes an instance's configuration (automations, skills, MCP server
 *     definitions, secret NAMES, LLM settings, agent profiles — never secret
 *     values, never conversations) into a `skin.yaml` + `automations/` layout,
 *   • ships an arbitrary web app started via `npm run start` on the single
 *     port given by the OPENHANDS_SKIN_PORT environment variable. The app
 *     must serve its UI under /skin/ (index.html + static, with
 *     <base href="/skin/">) and its own backend under /skin/api/* — the
 *     host proxies /skin verbatim and the Canvas frontend embeds it as
 *     its default tab (nested inside the Canvas UI, not standalone at /),
 *   • surfaces in the Canvas UI as an auto-created menu item (iframe tab).
 *
 * At most ONE skin can be installed per instance.
 *
 * The skin checkout lives in the agent's workspace (~/workspace/skin, a
 * persistent volume) so the instance's agent can edit the running skin in
 * place; this service supervises the app (own process group, restart with
 * backoff) and exposes POST /skin-api/restart for applying edits. Service
 * state (settings, logs) stays in ~/.openhands/agent-canvas/skin.
 *
 * This module is consumed by scripts/static-server.mjs, which mounts the
 * REST API under /skin-api and reverse-proxies /skin → the running skin.
 *
 * Skin repo format (everything at the repo root):
 *   skin.yaml       — required. name, icon (lucide icon name for the nav
 *                     entry), screenshot, canvas_version, secrets,
 *                     mcp_servers, skills, llm, settings, theme (major
 *                     colors inherited by the whole Canvas UI).
 *   package.json    — required, must define a "start" script.
 *   SKILL.md        — required. Describes what the skin does (pages, APIs,
 *                     data sources, agent workflows). Synced on every
 *                     start/restart/pull into ~/.openhands/skills/ as a
 *                     legacy always-active skill so its full content is in
 *                     the agent's context at the start of every
 *                     conversation. Must NOT declare `triggers:`
 *                     frontmatter (that would demote it to on-keyword).
 *   automations/    — optional, one subdirectory per automation with its
 *                     definition (and Python code / tarball).
 */

import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { spawn } from "node:child_process";
import { homedir } from "node:os";
import { join } from "node:path";

import yaml from "js-yaml";

const SKIN_SETTINGS_FILE = "skin-settings.json";
const SKIN_REPO_DIR = "repo";
const SKIN_LOG_FILE = "skin.log";
const SKIN_YAML = "skin.yaml";
const RESTART_BACKOFF_MS = [1000, 2000, 5000, 10000, 30000];

function log(...args) {
  console.log("[skin-service]", ...args);
}

// ─────────────────────────────────────────────────────────────────────────────
// Skin theme (skin.yaml `theme:` block)
//
// A skin may declare a handful of major colors; the Canvas UI and the skin
// app both inherit them. Validation is strict (hex colors only, known keys
// only) because the values come from a user-supplied manifest and end up in
// a stylesheet. Derivation of the full Canvas variable set lives here so
// there is exactly one source of truth (the frontend applies the derived
// map verbatim; /skin-api/theme.css serves the same map to the skin app).
// ─────────────────────────────────────────────────────────────────────────────

// Lucide icon names are lowercase kebab-case (e.g. "activity",
// "bar-chart-3"). The value comes from a user-supplied manifest and ends up
// in a component lookup, so validate the shape strictly; anything else is
// dropped (the frontend falls back to its default skin icon).
const LUCIDE_ICON_NAME_RE = /^[a-z0-9]+(-[a-z0-9]+)*$/;

/** Validate a skin.yaml `icon:` value (a lucide icon name). Returns the
 * name or null — never an error, a bad icon must not block an install. */
export function sanitizeIconName(icon) {
  if (typeof icon !== "string") return null;
  const name = icon.trim().toLowerCase();
  return name.length <= 64 && LUCIDE_ICON_NAME_RE.test(name) ? name : null;
}

// Name (and filename stem) of the always-active agent skill a skin's
// SKILL.md is synced to. Fixed so (a) uninstall knows what to remove and
// (b) it can never collide with the skin-builder skill installed as "skin".
export const SKIN_APP_SKILL_NAME = "skin-app";

/**
 * Render a skin repo's SKILL.md as a legacy always-active agent skill.
 *
 * The repo file may carry AgentSkills frontmatter (name/description/…). We
 * strip it and emit our own minimal frontmatter WITHOUT `triggers:` — in the
 * legacy .md format that means trigger=None, i.e. the full content is
 * injected into the system prompt of every conversation (the whole point:
 * the agent always knows what this instance's skin does).
 */
export function renderSkinAppSkill(skillMd, skinName) {
  let body = skillMd;
  let description = null;
  const fm = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(skillMd);
  if (fm) {
    body = skillMd.slice(fm[0].length);
    try {
      const meta = yaml.load(fm[1]);
      if (meta && typeof meta.description === "string") {
        description = meta.description;
      }
    } catch {
      // Malformed frontmatter — drop it, keep the body.
    }
  }
  const frontmatter = {
    name: SKIN_APP_SKILL_NAME,
    description:
      description ||
      `What the "${skinName || "installed"}" skin of this instance does.`,
  };
  return `---\n${yaml.dump(frontmatter).trimEnd()}\n---\n\n${body.trim()}\n`;
}

const THEME_KEYS = [
  "accent",
  "background",
  "surface",
  "text",
  "muted",
  "border",
  "success",
  "warning",
  "danger",
];
const HEX_COLOR_RE = /^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/;

/** Whitelist + hex-validate a raw `theme:` block. Returns null when the
 * manifest has no usable theme. Unknown keys and non-hex values are
 * dropped (never an error — a bad theme must not block a skin install). */
export function sanitizeTheme(raw) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const theme = {};
  for (const key of THEME_KEYS) {
    const value = raw[key];
    if (typeof value === "string" && HEX_COLOR_RE.test(value.trim())) {
      theme[key] = value.trim().toLowerCase();
    }
  }
  return Object.keys(theme).length > 0 ? theme : null;
}

function hexToRgb(hex) {
  let h = hex.replace("#", "");
  if (h.length === 3) {
    h = h
      .split("")
      .map((c) => c + c)
      .join("");
  }
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function rgbToHex([r, g, b]) {
  const c = (v) => Math.round(v).toString(16).padStart(2, "0");
  return `#${c(r)}${c(g)}${c(b)}`;
}

/** Gamma-encoded per-channel lerp — same result as CSS
 * `color-mix(in srgb, a pct%, b)`, but emitted as a concrete hex value so
 * derived shades can be further converted (e.g. to HSL for HeroUI). */
const mix = (a, pct, b) => {
  const A = hexToRgb(a);
  const B = hexToRgb(b);
  const w = pct / 100;
  return rgbToHex(A.map((v, i) => v * w + B[i] * (1 - w)));
};

/** "H S% L%" channel string — the format HeroUI stores colors in
 * (`hsl(var(--heroui-*))`). */
export function hexToHslChannels(hex) {
  const [r, g, b] = hexToRgb(hex).map((v) => v / 255);
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  const d = max - min;
  let h = 0;
  let s = 0;
  if (d !== 0) {
    s = d / (1 - Math.abs(2 * l - 1));
    if (max === r) h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  const r2 = (n) => Math.round(n * 100) / 100;
  return `${r2(h)} ${r2(s * 100)}% ${r2(l * 100)}%`;
}

/** Expand the (partial) major-color theme into the full set of CSS custom
 * properties the Canvas UI is built on. Missing anchors fall back to each
 * other so a minimal `theme: {accent, background}` still works; shades in
 * between are derived by mixing the anchors. */
export function themeVars(theme) {
  const t = sanitizeTheme(theme);
  if (!t) return null;

  const background = t.background || t.surface || "#0b0e14";
  const surface =
    t.surface || (t.background ? mix(background, 82, "#ffffff") : "#21252f");
  const text = t.text || "#eef2f7";
  const muted = t.muted || mix(text, 55, background);
  const border = t.border || mix(muted, 45, background);

  /** @type {Record<string, string>} */
  const vars = {
    // Grey scale: anchors are the declared colors, steps between are mixes.
    "--cool-grey-50": mix(text, 60, "#ffffff"),
    "--cool-grey-100": text,
    "--cool-grey-200": mix(text, 85, muted),
    "--cool-grey-300": mix(text, 60, muted),
    "--cool-grey-400": muted,
    "--cool-grey-500": mix(muted, 75, border),
    "--cool-grey-600": mix(muted, 45, border),
    "--cool-grey-700": border,
    "--cool-grey-800": mix(border, 55, surface),
    "--cool-grey-900": mix(surface, 80, border),
    "--cool-grey-925": surface,
    "--cool-grey-950": background,
    "--cool-grey-975": mix(background, 60, "#000000"),
  };

  // HeroUI reads colors from its own --heroui-* HSL-channel variables, not
  // from our --cool-grey-* scale, so mirror the scale onto the same stop
  // positions the stock themes use (see src/themes/color-themes.ts).
  const g = (n) => hexToHslChannels(vars[`--cool-grey-${n}`]);
  Object.assign(vars, {
    "--heroui-background": g(950),
    "--heroui-background-foreground": g(50),
    "--heroui-foreground-50": g(975),
    "--heroui-foreground-100": g(950),
    "--heroui-foreground-200": g(925),
    "--heroui-foreground-300": g(900),
    "--heroui-foreground-400": g(800),
    "--heroui-foreground-500": g(700),
    "--heroui-foreground-600": g(600),
    "--heroui-foreground-700": g(500),
    "--heroui-foreground-800": g(400),
    "--heroui-foreground-900": g(300),
    "--heroui-foreground": g(300),
    "--heroui-content1": g(925),
    "--heroui-content1-foreground": g(100),
    "--heroui-content2": g(900),
    "--heroui-content2-foreground": g(200),
    "--heroui-content3": g(800),
    "--heroui-content3-foreground": g(300),
    "--heroui-content4": g(700),
    "--heroui-content4-foreground": g(400),
    "--heroui-default-50": g(975),
    "--heroui-default-100": g(950),
    "--heroui-default-200": g(925),
    "--heroui-default-300": g(900),
    "--heroui-default-400": g(800),
    "--heroui-default-500": g(700),
    "--heroui-default-600": g(600),
    "--heroui-default-700": g(500),
    "--heroui-default-800": g(400),
    "--heroui-default-900": g(300),
    "--heroui-default-foreground": g(50),
    "--heroui-default": g(800),
  });

  if (t.accent) {
    vars["--oh-color-primary"] = t.accent;
    vars["--oh-color-logo"] = t.accent;
    vars["--oh-accent"] = t.accent;
    vars["--oh-accent-foreground"] = background;
    vars["--oh-warning"] = t.warning || t.accent;
    vars["--oh-warning-foreground"] = background;
  } else if (t.warning) {
    vars["--oh-warning"] = t.warning;
    vars["--oh-warning-foreground"] = background;
  }
  if (t.success) {
    vars["--oh-color-success"] = t.success;
    vars["--oh-success"] = t.success;
    vars["--oh-success-foreground"] = background;
  }
  if (t.danger) {
    vars["--oh-color-danger"] = t.danger;
    vars["--oh-danger"] = t.danger;
  }
  return vars;
}

/** Render the derived variables as a stylesheet for the skin app
 * (`GET /skin-api/theme.css`). */
export function themeCss(theme) {
  const vars = themeVars(theme);
  if (!vars) return "/* no skin theme declared */\n:root {}\n";
  const lines = Object.entries(vars).map(([k, v]) => `  ${k}: ${v};`);
  return `:root {\n${lines.join("\n")}\n}\n`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Version range check (minimal semver subset: ">=1.7.0 <2.0.0", "1.8.x", "*")
// ─────────────────────────────────────────────────────────────────────────────

function parseVersion(v) {
  const m = /^v?(\d+)\.(\d+)\.(\d+)/.exec(String(v).trim());
  if (!m) return null;
  return [Number(m[1]), Number(m[2]), Number(m[3])];
}

function compareVersions(a, b) {
  for (let i = 0; i < 3; i += 1) {
    if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1;
  }
  return 0;
}

export function satisfiesCanvasVersion(current, range) {
  if (!range || range === "*") return true;
  const cur = parseVersion(current);
  if (!cur) return true; // dev builds ("dev", "unknown") are never blocked
  const clauses = String(range).trim().split(/\s+/);
  for (const clause of clauses) {
    const m = /^(>=|<=|>|<|=)?\s*v?(\d+\.\d+\.\d+)$/.exec(clause);
    if (!m) return true; // unparseable range — don't block install
    const op = m[1] || "=";
    const cmp = compareVersions(cur, parseVersion(m[2]));
    const ok =
      (op === ">=" && cmp >= 0) ||
      (op === "<=" && cmp <= 0) ||
      (op === ">" && cmp > 0) ||
      (op === "<" && cmp < 0) ||
      (op === "=" && cmp === 0);
    if (!ok) return false;
  }
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// git helpers
// ─────────────────────────────────────────────────────────────────────────────

function runGit(args, { cwd, token } = {}) {
  return new Promise((resolve) => {
    const gitArgs = [...args];
    if (token) {
      const basic = Buffer.from(`x-access-token:${token}`).toString("base64");
      gitArgs.unshift(
        "-c",
        `http.https://github.com/.extraheader=AUTHORIZATION: basic ${basic}`,
      );
    }
    const child = spawn("git", gitArgs, {
      cwd,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d) => {
      stdout += d;
    });
    child.stderr.on("data", (d) => {
      stderr += d;
    });
    child.on("close", (code) => resolve({ code, stdout, stderr }));
    child.on("error", (err) =>
      resolve({ code: -1, stdout, stderr: String(err) }),
    );
  });
}

function parseGitHubRepo(url) {
  const m =
    /^(?:https?:\/\/github\.com\/|git@github\.com:|github:)([^/]+)\/([^/.]+?)(?:\.git)?\/?$/.exec(
      String(url).trim(),
    );
  if (!m) return null;
  return { owner: m[1], repo: m[2] };
}

// ─────────────────────────────────────────────────────────────────────────────
// SkinService
// ─────────────────────────────────────────────────────────────────────────────

export class SkinService {
  /**
   * @param {object} opts
   * @param {number} opts.skinPort       Port the skin app must listen on
   *                                     (OPENHANDS_SKIN_PORT).
   * @param {string} [opts.home]         State dir (default
   *                                     ~/.openhands/agent-canvas/skin).
   * @param {string} [opts.workspaceDir] Skin checkout dir (default
   *                                     ~/workspace/skin). Lives in the
   *                                     agent's workspace so the agent can
   *                                     edit the running skin in place.
   * @param {string} [opts.agentServerUrl]  e.g. http://127.0.0.1:18000
   * @param {string} [opts.automationUrl]   e.g. http://127.0.0.1:18001
   * @param {string} [opts.sessionApiKey]   key for both backends
   * @param {string} [opts.canvasVersion]   running Agent Canvas version
   * @param {string} [opts.githubToken]     token for private repos / PRs
   */
  constructor(opts) {
    this.skinPort = opts.skinPort;
    this.home =
      opts.home || join(homedir(), ".openhands", "agent-canvas", "skin");
    this.workspaceDir =
      opts.workspaceDir || join(homedir(), "workspace", "skin");
    this.agentServerUrl = opts.agentServerUrl || null;
    this.automationUrl = opts.automationUrl || null;
    this.sessionApiKey = opts.sessionApiKey || null;
    this.canvasVersion = opts.canvasVersion || "dev";
    this.githubToken = opts.githubToken || process.env.GITHUB_TOKEN || null;
    this.child = null;
    this.restartCount = 0;
    this.stopping = false;
    this.lastError = null;
    mkdirSync(this.home, { recursive: true });
    this.migrateLegacyRepoDir();
  }

  /** The skin checkout lives in the agent's workspace (persistent volume)
   * so the agent can edit the running skin in place and restart it via
   * POST /skin-api/restart. */
  get repoDir() {
    return this.workspaceDir;
  }

  /** Installs from before the workspace move live in
   * ~/.openhands/agent-canvas/skin/repo — relocate them once. cpSync (not
   * rename): the workspace is typically a different volume. */
  migrateLegacyRepoDir() {
    const legacy = join(this.home, SKIN_REPO_DIR);
    if (
      !existsSync(join(legacy, SKIN_YAML)) ||
      existsSync(join(this.repoDir, SKIN_YAML))
    ) {
      return;
    }
    log(`Migrating skin checkout ${legacy} → ${this.repoDir}…`);
    try {
      rmSync(this.repoDir, { recursive: true, force: true });
      cpSync(legacy, this.repoDir, { recursive: true });
      rmSync(legacy, { recursive: true, force: true });
    } catch (err) {
      log(`Skin migration failed: ${err.message}`);
    }
  }

  get settingsPath() {
    return join(this.home, SKIN_SETTINGS_FILE);
  }

  loadSettings() {
    try {
      return JSON.parse(readFileSync(this.settingsPath, "utf-8"));
    } catch {
      return null;
    }
  }

  saveSettings(settings) {
    writeFileSync(this.settingsPath, JSON.stringify(settings, null, 2));
  }

  readSkinYaml() {
    try {
      return yaml.load(readFileSync(join(this.repoDir, SKIN_YAML), "utf-8"));
    } catch {
      return null;
    }
  }

  isInstalled() {
    return !!this.loadSettings() && existsSync(join(this.repoDir, SKIN_YAML));
  }

  async currentBranch() {
    const res = await runGit(["rev-parse", "--abbrev-ref", "HEAD"], {
      cwd: this.repoDir,
    });
    return res.code === 0 ? res.stdout.trim() : null;
  }

  // ── status ────────────────────────────────────────────────────────────────

  async status() {
    if (!this.isInstalled()) {
      return { installed: false, running: false };
    }
    const settings = this.loadSettings();
    const skin = this.readSkinYaml() || {};
    return {
      installed: true,
      name: skin.name || "Skin",
      icon: sanitizeIconName(skin.icon),
      screenshot: skin.screenshot || null,
      repoUrl: settings.repoUrl,
      branch: await this.currentBranch(),
      autoPush: !!settings.autoPush,
      running: !!this.child,
      path: this.repoDir,
      port: this.skinPort,
      canvasVersion: this.canvasVersion,
      canvasVersionRange: skin.canvas_version || null,
      secrets: Array.isArray(skin.secrets) ? skin.secrets : [],
      theme: sanitizeTheme(skin.theme),
      themeVars: themeVars(skin.theme),
      error: this.lastError,
    };
  }

  // ── install / uninstall ───────────────────────────────────────────────────

  async install({ repoUrl, ref, autoPush = true }) {
    if (this.isInstalled()) {
      throw new SkinError(
        409,
        "A skin is already installed. Uninstall it first — only one skin can be installed at a time.",
      );
    }
    if (!parseGitHubRepo(repoUrl)) {
      throw new SkinError(
        400,
        "Only GitHub repository URLs are supported (e.g. https://github.com/owner/repo).",
      );
    }

    rmSync(this.repoDir, { recursive: true, force: true });
    const cloneArgs = ["clone", "--depth", "1"];
    if (ref) cloneArgs.push("--branch", ref);
    cloneArgs.push(repoUrl, this.repoDir);
    const clone = await runGit(cloneArgs, { token: this.githubToken });
    if (clone.code !== 0) {
      throw new SkinError(502, `git clone failed: ${clone.stderr.trim()}`);
    }

    const validationError = this.validateRepo();
    if (validationError) {
      rmSync(this.repoDir, { recursive: true, force: true });
      throw new SkinError(422, validationError);
    }

    this.saveSettings({
      repoUrl,
      ref: ref || null,
      autoPush: !!autoPush,
      installedAt: new Date().toISOString(),
    });

    const applyReport = await this.applyConfiguration();
    await this.start();
    return { ...(await this.status()), applyReport };
  }

  validateRepo() {
    const skin = this.readSkinYaml();
    if (!skin || typeof skin !== "object") {
      return "skin.yaml is missing or invalid at the repository root.";
    }
    if (!skin.name) {
      return "skin.yaml must define a `name`.";
    }
    let pkg;
    try {
      pkg = JSON.parse(
        readFileSync(join(this.repoDir, "package.json"), "utf-8"),
      );
    } catch {
      return "package.json is missing or invalid at the repository root.";
    }
    if (!pkg.scripts?.start) {
      return "package.json must define a `start` script (the skin is launched with `npm run start`).";
    }
    if (!existsSync(join(this.repoDir, "SKILL.md"))) {
      return "SKILL.md is missing at the repository root. Every skin must ship a SKILL.md describing what it does; it is loaded into the agent's context at the start of every conversation.";
    }
    if (!satisfiesCanvasVersion(this.canvasVersion, skin.canvas_version)) {
      return `This skin requires Agent Canvas ${skin.canvas_version}, but this instance is running ${this.canvasVersion}.`;
    }
    return null;
  }

  async uninstall() {
    await this.stop();
    rmSync(this.repoDir, { recursive: true, force: true });
    rmSync(this.settingsPath, { force: true });
    rmSync(this.skillPath, { force: true });
    this.lastError = null;
    return { installed: false, running: false };
  }

  // ── run the skin app ─────────────────────────────────────────────────────

  /** Absolute path of the always-active agent skill synced from the skin
   * repo's SKILL.md. Lives in ~/.openhands/skills/ (a user-skills dir the
   * SDK scans on every conversation start); legacy .md format without
   * `triggers:` means the full content lands in the agent's system prompt. */
  get skillPath() {
    return join(homedir(), ".openhands", "skills", `${SKIN_APP_SKILL_NAME}.md`);
  }

  /** Sync the skin repo's SKILL.md → the always-active skill file. Called on
   * every start (install/restart/pull all funnel through start), so agent
   * edits to the skin's SKILL.md take effect on the next restart. */
  syncSkillMd() {
    const src = join(this.repoDir, "SKILL.md");
    try {
      if (!existsSync(src)) {
        // Required for new installs (validateRepo); tolerated for skins
        // installed before the requirement existed.
        rmSync(this.skillPath, { force: true });
        return;
      }
      const skin = this.readSkinYaml() || {};
      mkdirSync(join(homedir(), ".openhands", "skills"), { recursive: true });
      writeFileSync(
        this.skillPath,
        renderSkinAppSkill(readFileSync(src, "utf-8"), skin.name),
      );
      log(`Synced skin SKILL.md → ${this.skillPath}`);
    } catch (err) {
      log(`Failed to sync skin SKILL.md: ${err.message}`);
    }
  }

  async start() {
    if (!this.isInstalled() || this.child) return;
    this.stopping = false;
    this.syncSkillMd();

    const pkg = JSON.parse(
      readFileSync(join(this.repoDir, "package.json"), "utf-8"),
    );
    if (pkg.dependencies || pkg.devDependencies) {
      log("Running npm install for skin dependencies…");
      await new Promise((resolve) => {
        const npmInstall = spawn(
          "npm",
          ["install", "--omit=dev", "--no-audit", "--no-fund"],
          { cwd: this.repoDir, stdio: "inherit" },
        );
        npmInstall.on("close", resolve);
        npmInstall.on("error", resolve);
      });
    }

    this.spawnSkin();
  }

  spawnSkin() {
    if (this.stopping) return;
    log(`Starting skin app on port ${this.skinPort} (npm run start)…`);
    const logPath = join(this.home, SKIN_LOG_FILE);
    // detached: the skin runs in its own process group so stop() can kill
    // the whole tree (npm + the actual server), never leaving an orphan
    // holding the port.
    const child = spawn("npm", ["run", "start"], {
      cwd: this.repoDir,
      detached: true,
      env: {
        ...process.env,
        OPENHANDS_SKIN_PORT: String(this.skinPort),
        PORT: String(this.skinPort),
        AGENT_SERVER_URL: this.agentServerUrl || "",
        AUTOMATION_URL: this.automationUrl || "",
        SESSION_API_KEY: this.sessionApiKey || "",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    const append = (d) => {
      try {
        writeFileSync(logPath, d, { flag: "a" });
      } catch {
        /* best effort */
      }
    };
    child.stdout.on("data", append);
    child.stderr.on("data", append);
    child.on("close", (code) => {
      this.child = null;
      if (this.stopping) return;
      const backoff =
        RESTART_BACKOFF_MS[
          Math.min(this.restartCount, RESTART_BACKOFF_MS.length - 1)
        ];
      this.restartCount += 1;
      this.lastError = `Skin app exited with code ${code}; restarting in ${backoff}ms`;
      log(this.lastError);
      setTimeout(() => this.spawnSkin(), backoff);
    });
    child.on("error", (err) => {
      this.child = null;
      this.lastError = `Failed to start skin app: ${err.message}`;
      log(this.lastError);
    });
    this.child = child;
  }

  async stop() {
    this.stopping = true;
    const child = this.child;
    if (!child) return;
    this.child = null;
    const exited = new Promise((resolve) => {
      child.once("close", resolve);
      setTimeout(resolve, 5000).unref?.();
    });
    try {
      process.kill(-child.pid, "SIGTERM");
    } catch {
      child.kill("SIGTERM");
    }
    await exited;
    // Whatever survived SIGTERM gets the axe — a lingering server would
    // hold the port and make the restarted skin crash-loop on EADDRINUSE.
    try {
      process.kill(-child.pid, "SIGKILL");
    } catch {
      /* group already gone */
    }
  }

  /** Cleanly restart the skin app (e.g. after the agent edited the code in
   * the workspace checkout). Re-runs npm install when needed. */
  async restart() {
    this.requireInstalled();
    await this.stop();
    this.restartCount = 0;
    this.lastError = null;
    await this.start();
    return this.status();
  }

  // ── pull / push / auto-push ──────────────────────────────────────────────

  async pull() {
    this.requireInstalled();
    const res = await runGit(["pull", "--ff-only"], {
      cwd: this.repoDir,
      token: this.githubToken,
    });
    if (res.code !== 0) {
      throw new SkinError(502, `git pull failed: ${res.stderr.trim()}`);
    }
    await this.stop();
    await this.start();
    return this.status();
  }

  /**
   * Push local skin changes upstream. Prefers pushing straight to the
   * currently checked-out branch (main by default). If push access is
   * missing, falls back to pushing a new branch and opening a pull request.
   */
  async push({ message } = {}) {
    this.requireInstalled();
    const commitMessage = message || "Update skin (agent-modified)";

    await runGit(["add", "-A"], { cwd: this.repoDir });
    const commit = await runGit(["commit", "-m", commitMessage], {
      cwd: this.repoDir,
    });
    const nothingToCommit =
      commit.code !== 0 && /nothing to commit/i.test(commit.stdout);
    if (commit.code !== 0 && !nothingToCommit) {
      throw new SkinError(500, `git commit failed: ${commit.stderr.trim()}`);
    }

    const branch = (await this.currentBranch()) || "main";
    const push = await runGit(["push", "origin", `HEAD:${branch}`], {
      cwd: this.repoDir,
      token: this.githubToken,
    });
    if (push.code === 0) {
      return { pushed: true, branch, pullRequest: null };
    }

    // No push access to the branch — push a topic branch and open a PR.
    const settings = this.loadSettings();
    const gh = parseGitHubRepo(settings.repoUrl);
    const topic = `skin-update-${Date.now()}`;
    const pushTopic = await runGit(["push", "origin", `HEAD:${topic}`], {
      cwd: this.repoDir,
      token: this.githubToken,
    });
    if (pushTopic.code !== 0 || !gh || !this.githubToken) {
      throw new SkinError(
        502,
        `git push failed (no access to ${branch}, topic-branch fallback also failed): ${push.stderr.trim()}`,
      );
    }
    const pr = await this.openPullRequest(gh, topic, branch, commitMessage);
    return { pushed: true, branch: topic, pullRequest: pr };
  }

  async openPullRequest(gh, head, base, title) {
    const res = await fetch(
      `https://api.github.com/repos/${gh.owner}/${gh.repo}/pulls`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.githubToken}`,
          Accept: "application/vnd.github+json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          title,
          head,
          base,
          body: "Automated skin update pushed from an Agent Canvas instance.\n\n_This pull request was created by an AI agent (OpenHands)._",
        }),
      },
    );
    if (!res.ok) return null;
    const data = await res.json();
    return { url: data.html_url, number: data.number };
  }

  async setAutoPush(autoPush) {
    this.requireInstalled();
    const settings = this.loadSettings();
    settings.autoPush = !!autoPush;
    this.saveSettings(settings);
    return this.status();
  }

  /** Called after the agent modifies the skin; honors the auto-push flag. */
  async autoPushIfEnabled(message) {
    const settings = this.loadSettings();
    if (!settings?.autoPush) return { pushed: false };
    try {
      return await this.push({ message });
    } catch (err) {
      // Auto-push failures must never block the skin from running.
      this.lastError = `Auto-push failed: ${err.message}`;
      log(this.lastError);
      return { pushed: false, error: err.message };
    }
  }

  requireInstalled() {
    if (!this.isInstalled()) {
      throw new SkinError(409, "No skin is installed.");
    }
  }

  // ── backend fetch helpers ─────────────────────────────────────────────────

  async backendFetch(base, path, init = {}) {
    const res = await fetch(`${base}${path}`, {
      ...init,
      headers: {
        "X-Session-API-Key": this.sessionApiKey || "",
        "Content-Type": "application/json",
        ...(init.headers || {}),
      },
    });
    if (!res.ok) {
      throw new Error(`${path} -> HTTP ${res.status}`);
    }
    const contentType = res.headers.get("content-type") || "";
    return contentType.includes("json") ? res.json() : res.arrayBuffer();
  }

  // ── serialization (export instance config → skin repo) ──────────────────
  //
  // Serialized:   automations (+code), skills, MCP server definitions,
  //               secret NAMES, LLM settings, agent profiles, settings.
  // NEVER:        conversations, secret VALUES, credential material.

  async exportConfiguration() {
    this.requireInstalled();
    if (!this.agentServerUrl) {
      throw new SkinError(503, "Agent server URL is not configured.");
    }

    const skin = this.readSkinYaml() || {};
    const report = { exported: [], skipped: [] };

    // Settings (masked — the agent server never returns raw secret values
    // without X-Expose-Secrets, which we deliberately do not send).
    let settings = null;
    try {
      settings = await this.backendFetch(this.agentServerUrl, "/api/settings");
    } catch (err) {
      report.skipped.push(`settings: ${err.message}`);
    }
    if (settings) {
      const agentSettings = settings.agent_settings || {};
      skin.mcp_servers = stripSecretValues(agentSettings.mcp_config || {});
      skin.llm = stripSecretValues({
        ...(agentSettings.llm || {}),
        api_key: undefined,
      });
      skin.settings = stripSecretValues(settings.conversation_settings || {});
      report.exported.push("settings", "mcp_servers", "llm");
    }

    // Agent profiles.
    try {
      const profiles = await this.backendFetch(
        this.agentServerUrl,
        "/api/agent-profiles",
      );
      skin.agent_profiles = (profiles.profiles || profiles || []).map((p) =>
        stripSecretValues(p),
      );
      report.exported.push("agent_profiles");
    } catch (err) {
      report.skipped.push(`agent_profiles: ${err.message}`);
    }

    // Secret NAMES only.
    try {
      const secrets = await this.backendFetch(
        this.agentServerUrl,
        "/api/settings/secrets",
      );
      skin.secrets = (secrets.secrets || []).map((s) => ({
        name: s.name,
        description: s.description || "",
      }));
      report.exported.push("secrets (names only)");
    } catch (err) {
      report.skipped.push(`secrets: ${err.message}`);
    }

    // Skills (source references).
    try {
      const skills = await this.backendFetch(
        this.agentServerUrl,
        "/api/skills",
      );
      skin.skills = (skills.skills || [])
        .filter((s) => s.source)
        .map((s) => ({ name: s.name, source: s.source }));
      report.exported.push("skills");
    } catch (err) {
      report.skipped.push(`skills: ${err.message}`);
    }

    // Automations: definition + tarball (contains the Python code).
    if (this.automationUrl) {
      try {
        const list = await this.backendFetch(
          this.automationUrl,
          "/api/automation/v1?limit=100",
        );
        const automations = list.items || list.automations || [];
        const automationsDir = join(this.repoDir, "automations");
        mkdirSync(automationsDir, { recursive: true });
        for (const automation of automations) {
          const dir = join(automationsDir, sanitizeName(automation.name));
          mkdirSync(dir, { recursive: true });
          writeFileSync(
            join(dir, "automation.json"),
            JSON.stringify(stripSecretValues(automation), null, 2),
          );
          try {
            const tarball = await this.backendFetch(
              this.automationUrl,
              `/api/automation/v1/${encodeURIComponent(automation.id)}/tarball`,
            );
            writeFileSync(join(dir, "automation.tar.gz"), Buffer.from(tarball));
          } catch (err) {
            report.skipped.push(
              `automation tarball ${automation.name}: ${err.message}`,
            );
          }
        }
        report.exported.push(`automations (${automations.length})`);
      } catch (err) {
        report.skipped.push(`automations: ${err.message}`);
      }
    }

    skin.canvas_version = skin.canvas_version || `>=${this.canvasVersion}`;
    writeFileSync(join(this.repoDir, SKIN_YAML), yaml.dump(skin));

    const pushResult = await this.autoPushIfEnabled(
      "Export Agent Canvas instance configuration to skin",
    );
    return { report, pushResult };
  }

  // ── deserialization (apply skin config → this instance) ─────────────────

  async applyConfiguration() {
    const skin = this.readSkinYaml() || {};
    const report = { applied: [], skipped: [] };
    if (!this.agentServerUrl) {
      report.skipped.push("no agent server URL configured");
      return report;
    }

    // MCP servers + LLM settings + conversation settings.
    const agentSettingsDiff = {};
    if (skin.mcp_servers && Object.keys(skin.mcp_servers).length > 0) {
      agentSettingsDiff.mcp_config = skin.mcp_servers;
    }
    if (skin.llm && Object.keys(skin.llm).length > 0) {
      agentSettingsDiff.llm = skin.llm;
    }
    if (Object.keys(agentSettingsDiff).length > 0) {
      try {
        await this.backendFetch(this.agentServerUrl, "/api/settings", {
          method: "PATCH",
          body: JSON.stringify({
            agent_settings_diff: agentSettingsDiff,
            ...(skin.settings
              ? { conversation_settings_diff: skin.settings }
              : {}),
          }),
        });
        report.applied.push("settings (mcp_servers, llm)");
      } catch (err) {
        report.skipped.push(`settings: ${err.message}`);
      }
    }

    // Skills.
    for (const skill of skin.skills || []) {
      try {
        await this.backendFetch(this.agentServerUrl, "/api/skills/install", {
          method: "POST",
          body: JSON.stringify({ source: skill.source, force: true }),
        });
        report.applied.push(`skill ${skill.name}`);
      } catch (err) {
        report.skipped.push(`skill ${skill.name}: ${err.message}`);
      }
    }

    // Agent profiles.
    for (const profile of skin.agent_profiles || []) {
      if (!profile.name) continue;
      try {
        await this.backendFetch(
          this.agentServerUrl,
          `/api/agent-profiles/${encodeURIComponent(profile.name)}`,
          { method: "POST", body: JSON.stringify(profile) },
        );
        report.applied.push(`agent profile ${profile.name}`);
      } catch (err) {
        report.skipped.push(`agent profile ${profile.name}: ${err.message}`);
      }
    }

    // Automations are re-registered from their stored tarballs when the
    // automation backend supports tarball import; report otherwise.
    const automationsDir = join(this.repoDir, "automations");
    if (existsSync(automationsDir)) {
      report.skipped.push(
        "automations: bundled in repo (automations/); re-register via the Automations UI or the automation API",
      );
    }

    return report;
  }
}

export class SkinError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function sanitizeName(name) {
  return String(name)
    .toLowerCase()
    .replace(/[^a-z0-9-_]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
}

/**
 * Recursively strip anything that looks like credential material. Masked
 * values from the agent server ("**********") and common secret-ish keys are
 * removed so serialized configs never carry secret values — only names.
 */
export function stripSecretValues(value) {
  if (Array.isArray(value)) return value.map(stripSecretValues);
  if (!value || typeof value !== "object") return value;
  const SECRETISH = /(api_?key|token|secret|password|credential)/i;
  const out = {};
  for (const [key, v] of Object.entries(value)) {
    if (v === undefined) continue;
    if (typeof v === "string" && /^\*{4,}$/.test(v)) continue;
    if (SECRETISH.test(key) && typeof v === "string" && v.length > 0) continue;
    out[key] = stripSecretValues(v);
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP handler — mounted by static-server.mjs under /skin-api
// ─────────────────────────────────────────────────────────────────────────────

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk) => {
      data += chunk;
      if (data.length > 1_000_000) reject(new Error("Body too large"));
    });
    req.on("end", () => {
      try {
        resolve(data ? JSON.parse(data) : {});
      } catch {
        reject(new SkinError(400, "Invalid JSON body"));
      }
    });
    req.on("error", reject);
  });
}

function send(res, status, body) {
  const data = JSON.stringify(body);
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Cache-Control": "no-store",
  });
  res.end(data);
}

/**
 * Create an HTTP handler for /skin-api requests.
 * Mutating requests require the session API key (X-Session-API-Key header),
 * matching how the agent-server and automation backends authenticate.
 */
export function createSkinApiHandler(service) {
  return async function handleSkinApi(req, res, pathname) {
    try {
      const method = req.method || "GET";
      const isThemeRead =
        method === "GET" && pathname === "/skin-api/theme.css";
      if (isThemeRead) {
        // Public like /status: it's a <link> from the skin app, which can't
        // send auth headers. Contains only sanitized colors, no secrets.
        const skin = (service.isInstalled() && service.readSkinYaml()) || {};
        const css = themeCss(skin.theme);
        res.writeHead(200, {
          "Content-Type": "text/css; charset=utf-8",
          "Content-Length": Buffer.byteLength(css),
          "Cache-Control": "no-cache",
        });
        res.end(css);
        return;
      }
      const isStatusRead = method === "GET" && pathname === "/skin-api/status";
      if (!isStatusRead && service.sessionApiKey) {
        const provided = req.headers["x-session-api-key"];
        if (provided !== service.sessionApiKey) {
          send(res, 401, { error: "Invalid or missing X-Session-API-Key" });
          return;
        }
      }

      if (isStatusRead) {
        send(res, 200, await service.status());
        return;
      }
      if (method === "POST" && pathname === "/skin-api/install") {
        const body = await readBody(req);
        send(res, 200, await service.install(body));
        return;
      }
      if (method === "POST" && pathname === "/skin-api/uninstall") {
        send(res, 200, await service.uninstall());
        return;
      }
      if (method === "POST" && pathname === "/skin-api/pull") {
        send(res, 200, await service.pull());
        return;
      }
      if (method === "POST" && pathname === "/skin-api/restart") {
        send(res, 200, await service.restart());
        return;
      }
      if (method === "POST" && pathname === "/skin-api/push") {
        const body = await readBody(req);
        send(res, 200, await service.push(body));
        return;
      }
      if (method === "POST" && pathname === "/skin-api/export") {
        send(res, 200, await service.exportConfiguration());
        return;
      }
      if (method === "PATCH" && pathname === "/skin-api/settings") {
        const body = await readBody(req);
        send(res, 200, await service.setAutoPush(body.autoPush));
        return;
      }
      send(res, 404, { error: "Not found" });
    } catch (err) {
      const status = err instanceof SkinError ? err.status : 500;
      send(res, status, { error: err.message });
    }
  };
}
