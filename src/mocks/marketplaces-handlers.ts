import { http, HttpResponse } from "msw";
import { SKILLS_CATALOG } from "@openhands/extensions/skills";
import { INTEGRATION_CATALOG } from "@openhands/extensions/integrations";
import { AUTOMATION_CATALOG } from "@openhands/extensions/automations";
import { APPS_CATALOG } from "@openhands/extensions/apps";
import openhandsMarketplace from "@openhands/extensions/marketplaces/openhands-extensions.json";
import knownSources from "#/fixtures/marketplaces/known-sources.json";
import type {
  AddMarketplaceRequest,
  CatalogEntry,
  CatalogEntryIcon,
  CatalogEntryInstall,
  CatalogError,
  CatalogKind,
  MarketplaceCatalog,
  MarketplaceSource,
  UpdateMarketplaceRequest,
} from "#/types/marketplaces";
import { CATALOG_KINDS } from "#/types/marketplaces";

/**
 * Browser-development implementation of the marketplaces API.
 *
 * The default marketplace is built from the `@openhands/extensions` package
 * Canvas already ships, so it shows the real catalog. Two more sources are
 * known by their `source` string (see `src/fixtures/marketplaces`), and any
 * other well-formed source gets a small generated catalog, so the add flow can
 * be exercised end to end. State lives in `sessionStorage` so a reload keeps
 * what was added, installed and trusted.
 */

export const DEFAULT_MARKETPLACE_NAME = "openhands-extensions";
export const DEFAULT_MARKETPLACE_SOURCE = "github:OpenHands/extensions";

const STORAGE_KEY = "openhands-marketplaces-dev-state";

interface SeedEntry {
  kind: CatalogKind;
  name: string;
  description: string;
  tags: string[];
  version?: string;
  runs_code: boolean;
  transport?: "stdio" | "http";
  requires?: string[];
  icon?: CatalogEntryIcon;
}

interface KnownSource {
  source: string;
  ref: string;
  resolved_ref: string;
  entries: SeedEntry[];
  errors: CatalogError[];
}

interface StoredMarketplace {
  source: Omit<MarketplaceSource, "counts" | "error_count">;
  entries: SeedEntry[];
  errors: CatalogError[];
}

interface StoredState {
  marketplaces: StoredMarketplace[];
  /** Keyed by entry id. */
  installs: Record<string, CatalogEntryInstall>;
}

const KNOWN_SOURCES = knownSources as Record<string, KnownSource>;

// ---------------------------------------------------------------------------
// Seed data
// ---------------------------------------------------------------------------

function defaultMarketplaceEntries(): SeedEntry[] {
  const plugins: SeedEntry[] = openhandsMarketplace.plugins
    .filter((plugin) => String(plugin.source).startsWith("./plugins/"))
    .map((plugin) => ({
      kind: "plugin",
      name: plugin.name,
      description: plugin.description ?? "",
      tags: [plugin.category, ...(plugin.keywords ?? [])].filter(
        (tag): tag is string => typeof tag === "string" && tag.length > 0,
      ),
      runs_code: true,
    }));

  const skills: SeedEntry[] = SKILLS_CATALOG.map((skill) => ({
    kind: "skill",
    name: skill.name,
    description: skill.description,
    tags: [skill.category, ...skill.triggers.slice(0, 3)],
    runs_code: false,
  }));

  // The published catalog only lists remote (shttp/sse) servers; a server
  // started as a local command would run code, like acme-vault-cli does.
  const mcp: SeedEntry[] = INTEGRATION_CATALOG.map((integration) => ({
    kind: "mcp",
    name: integration.id,
    description: integration.description,
    tags: [...(integration.categories ?? []), ...(integration.keywords ?? [])]
      .map((tag) => tag.toLowerCase())
      .slice(0, 5),
    runs_code: false,
    transport: "http",
    icon: {
      logo_url: integration.logoUrl ?? null,
      bg: integration.iconBg ?? null,
      color: integration.iconColor ?? null,
      slug: null,
    },
  }));

  const apps: SeedEntry[] = APPS_CATALOG.map((app) => ({
    kind: "app",
    name: app.id,
    description: app.description,
    tags: app.beta ? ["beta"] : [],
    runs_code: true,
  }));

  const automations: SeedEntry[] = AUTOMATION_CATALOG.map((automation) => ({
    kind: "automation",
    name: automation.id,
    description: automation.description,
    tags: [automation.category, ...(automation.triggers ?? [])],
    version: automation.version,
    runs_code: Boolean(automation.setup?.bundle),
    requires: Object.keys(automation.requires.integrations),
    icon: automation.icon
      ? { logo_url: null, bg: null, color: null, slug: automation.icon }
      : undefined,
  }));

  return [...mcp, ...skills, ...plugins, ...apps, ...automations].map(
    (entry) => ({
      ...entry,
      tags: Array.from(new Set(entry.tags.map((tag) => tag.toLowerCase()))),
    }),
  );
}

function nowIso(): string {
  return new Date().toISOString();
}

function seedState(): StoredState {
  return {
    marketplaces: [
      {
        source: {
          name: DEFAULT_MARKETPLACE_NAME,
          source: DEFAULT_MARKETPLACE_SOURCE,
          ref: "main",
          repo_path: null,
          scope: "instance",
          trusted: true,
          token_secret: null,
          is_default: true,
          resolved_ref: "fb020c71",
          last_refreshed_at: nowIso(),
        },
        entries: defaultMarketplaceEntries(),
        errors: [],
      },
      {
        source: {
          name: "acme-internal",
          source: KNOWN_SOURCES["acme-internal"].source,
          ref: KNOWN_SOURCES["acme-internal"].ref,
          repo_path: null,
          scope: "user",
          trusted: false,
          token_secret: "ACME_GITHUB_TOKEN",
          is_default: false,
          resolved_ref: KNOWN_SOURCES["acme-internal"].resolved_ref,
          last_refreshed_at: nowIso(),
        },
        entries: KNOWN_SOURCES["acme-internal"].entries,
        errors: KNOWN_SOURCES["acme-internal"].errors,
      },
    ],
    installs: {},
  };
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

function readState(): StoredState | null {
  try {
    const value = globalThis.sessionStorage?.getItem(STORAGE_KEY);
    return value ? (JSON.parse(value) as StoredState) : null;
  } catch {
    return null;
  }
}

let state: StoredState = readState() ?? seedState();

function persist(): void {
  try {
    globalThis.sessionStorage?.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // The module variable still provides an in-memory fallback.
  }
}

export function resetMarketplacesMockData(): void {
  state = seedState();
  try {
    globalThis.sessionStorage?.removeItem(STORAGE_KEY);
  } catch {
    // Nothing persisted to clear.
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function entryId(kind: CatalogKind, name: string, marketplace: string): string {
  return `${kind}:${name}@${marketplace}`;
}

function emptyCounts(): Record<CatalogKind, number> {
  return { mcp: 0, skill: 0, plugin: 0, app: 0, automation: 0 };
}

function toSource(stored: StoredMarketplace): MarketplaceSource {
  const counts = emptyCounts();
  for (const entry of stored.entries) counts[entry.kind] += 1;
  return { ...stored.source, counts, error_count: stored.errors.length };
}

function findMarketplace(name: string): StoredMarketplace | undefined {
  return state.marketplaces.find((m) => m.source.name === name);
}

function notInstalled(): CatalogEntryInstall {
  return {
    state: "not_installed",
    enabled: false,
    installed_ref: null,
    policy: "track",
  };
}

function toEntry(seed: SeedEntry, stored: StoredMarketplace): CatalogEntry {
  const id = entryId(seed.kind, seed.name, stored.source.name);
  const alsoIn = state.marketplaces
    .filter(
      (other) =>
        other.source.name !== stored.source.name &&
        other.entries.some((e) => e.kind === seed.kind && e.name === seed.name),
    )
    .map((other) => other.source.name);
  return {
    id,
    kind: seed.kind,
    name: seed.name,
    marketplace: stored.source.name,
    description: seed.description,
    tags: seed.tags,
    version: seed.version ?? null,
    runs_code: seed.runs_code,
    also_in: alsoIn,
    install: state.installs[id] ?? notInstalled(),
    requires: seed.requires ?? [],
    transport: seed.transport ?? null,
    icon: seed.icon ?? null,
  };
}

function toCatalog(stored: StoredMarketplace): MarketplaceCatalog {
  return {
    marketplace: toSource(stored),
    entries: stored.entries.map((seed) => toEntry(seed, stored)),
    errors: stored.errors,
  };
}

function getParam(value: string | readonly string[] | undefined): string {
  if (!value) return "";
  return decodeURIComponent(Array.isArray(value) ? value[0] : value);
}

function detail(message: string, status: number) {
  return HttpResponse.json({ detail: message }, { status });
}

function marketplaceNotFound(name: string) {
  return detail(`Marketplace '${name}' is not registered.`, 404);
}

/** `github:owner/repo` -> `repo`; URLs and paths use their last segment. */
function deriveName(source: string): string {
  const trimmed = source
    .trim()
    .replace(/\.git$/, "")
    .replace(/\/+$/, "");
  const last = trimmed.split(/[/:]/).filter(Boolean).pop() ?? "marketplace";
  return last
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function looksLikeSource(source: string): boolean {
  return (
    /^github:[^/\s]+\/[^/\s]+/.test(source) ||
    /^(https?|ssh|git):\/\/\S+/.test(source) ||
    /^git@\S+:\S+/.test(source) ||
    source.startsWith("/") ||
    source.startsWith("~") ||
    source.startsWith("./") ||
    source.startsWith("file://")
  );
}

function fakeCommit(seed: string): string {
  let hash = 2166136261;
  for (const char of seed) {
    hash ^= char.charCodeAt(0);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash.toString(16).padStart(8, "0").slice(0, 8);
}

/** A believable catalog for a source the prototype does not know. */
function generatedEntries(name: string): SeedEntry[] {
  return [
    {
      kind: "skill",
      name: `${name}-conventions`,
      description: `Coding conventions and review rules kept in ${name}.`,
      tags: ["conventions"],
      runs_code: false,
    },
    {
      kind: "skill",
      name: `${name}-runbook`,
      description: `Operational runbook for the services ${name} covers.`,
      tags: ["operations"],
      runs_code: false,
    },
    {
      kind: "plugin",
      name: `${name}-tools`,
      description: `Slash commands and hooks shipped by ${name}.`,
      tags: ["hooks"],
      version: "0.1.0",
      runs_code: true,
    },
    {
      kind: "mcp",
      name: `${name}-api`,
      description: `MCP server for the ${name} HTTP API.`,
      tags: ["api"],
      runs_code: false,
      transport: "http",
    },
    {
      kind: "automation",
      name: `${name}-nightly-report`,
      description: `Nightly summary of activity across ${name} repositories.`,
      tags: ["scheduled"],
      runs_code: false,
      requires: ["github"],
    },
  ];
}

function resolveKnownSource(source: string): KnownSource | null {
  const normalized = source
    .trim()
    .toLowerCase()
    .replace(/\.git$/, "");
  for (const known of Object.values(KNOWN_SOURCES)) {
    if (known.source.toLowerCase() === normalized) return known;
    const path = known.source.replace(/^github:/, "").toLowerCase();
    if (normalized === `https://github.com/${path}`) return known;
  }
  if (
    normalized === DEFAULT_MARKETPLACE_SOURCE.toLowerCase() ||
    normalized === "https://github.com/openhands/extensions"
  ) {
    return {
      source: DEFAULT_MARKETPLACE_SOURCE,
      ref: "main",
      resolved_ref: "fb020c71",
      entries: defaultMarketplaceEntries(),
      errors: [],
    };
  }
  return null;
}

function installFrom(stored: StoredMarketplace): CatalogEntryInstall {
  return {
    state: "installed",
    enabled: true,
    installed_ref: stored.source.resolved_ref,
    policy: "track",
  };
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

export const MARKETPLACES_HANDLERS = [
  http.get("*/api/marketplaces", () =>
    HttpResponse.json({ marketplaces: state.marketplaces.map(toSource) }),
  ),

  http.post("*/api/marketplaces", async ({ request }) => {
    const body = (await request.json()) as AddMarketplaceRequest;
    const source = (body.source ?? "").trim();
    if (!looksLikeSource(source)) {
      return detail(
        `Unable to parse marketplace source: ${source || "(empty)"}. ` +
          "Use github:owner/repo, a git URL or a local path.",
        400,
      );
    }
    if (/private/i.test(source) && !body.token_secret) {
      return detail(
        "Authentication required: the repository is private. " +
          "Add an access token and try again.",
        401,
      );
    }
    const name = (body.name ?? "").trim() || deriveName(source);
    if (findMarketplace(name)) {
      return detail(`A marketplace named '${name}' already exists.`, 409);
    }

    const known = resolveKnownSource(source);
    const ref = (body.ref ?? "").trim() || known?.ref || "main";
    const stored: StoredMarketplace = {
      source: {
        name,
        source,
        ref,
        repo_path: null,
        scope: "user",
        trusted: Boolean(body.trusted),
        token_secret: body.token_secret?.trim() || null,
        is_default: false,
        resolved_ref: known?.resolved_ref ?? fakeCommit(`${source}@${ref}`),
        last_refreshed_at: nowIso(),
      },
      entries: known?.entries ?? generatedEntries(name),
      errors: known?.errors ?? [],
    };
    state.marketplaces.push(stored);
    persist();
    return HttpResponse.json(toSource(stored), { status: 201 });
  }),

  http.patch("*/api/marketplaces/:name", async ({ params, request }) => {
    const name = getParam(params.name);
    const stored = findMarketplace(name);
    if (!stored) return marketplaceNotFound(name);
    const body = (await request.json()) as UpdateMarketplaceRequest;
    if (typeof body.trusted === "boolean") {
      stored.source.trusted = body.trusted;
    }
    persist();
    return HttpResponse.json(toSource(stored));
  }),

  http.delete("*/api/marketplaces/:name", ({ params }) => {
    const name = getParam(params.name);
    const stored = findMarketplace(name);
    if (!stored) return marketplaceNotFound(name);
    if (stored.source.is_default) {
      return detail(
        `'${name}' is the instance default and cannot be removed.`,
        403,
      );
    }
    state.marketplaces = state.marketplaces.filter((m) => m !== stored);
    persist();
    return new HttpResponse(null, { status: 204 });
  }),

  http.post("*/api/marketplaces/:name/refresh", ({ params }) => {
    const name = getParam(params.name);
    const stored = findMarketplace(name);
    if (!stored) return marketplaceNotFound(name);
    // The ref moved: tracked installs from this marketplace now have an update.
    const previous = stored.source.resolved_ref ?? "";
    stored.source.resolved_ref = fakeCommit(`${previous}:${Date.now()}`);
    stored.source.last_refreshed_at = nowIso();
    for (const seed of stored.entries) {
      const id = entryId(seed.kind, seed.name, name);
      const install = state.installs[id];
      if (install?.state === "installed" && install.policy === "track") {
        state.installs[id] = { ...install, state: "update_available" };
      }
    }
    persist();
    return HttpResponse.json(toSource(stored));
  }),

  http.get("*/api/marketplaces/:name/catalog", ({ params }) => {
    const name = getParam(params.name);
    const stored = findMarketplace(name);
    if (!stored) return marketplaceNotFound(name);
    return HttpResponse.json(toCatalog(stored));
  }),

  http.post(
    "*/api/marketplaces/:name/entries/:entryId/install",
    ({ params }) => {
      const name = getParam(params.name);
      const id = getParam(params.entryId);
      const stored = findMarketplace(name);
      if (!stored) return marketplaceNotFound(name);
      const seed = stored.entries.find(
        (e) => entryId(e.kind, e.name, name) === id,
      );
      if (!seed) return detail(`Entry '${id}' is not in '${name}'.`, 404);
      if (seed.runs_code && !stored.source.trusted) {
        return detail(
          `'${seed.name}' runs code (hooks, bundles or a local command). ` +
            `Mark '${name}' as trusted before installing it.`,
          403,
        );
      }
      state.installs[id] = installFrom(stored);
      persist();
      return HttpResponse.json(toEntry(seed, stored));
    },
  ),

  http.post(
    "*/api/marketplaces/:name/entries/:entryId/update",
    ({ params }) => {
      const name = getParam(params.name);
      const id = getParam(params.entryId);
      const stored = findMarketplace(name);
      if (!stored) return marketplaceNotFound(name);
      const seed = stored.entries.find(
        (e) => entryId(e.kind, e.name, name) === id,
      );
      if (!seed) return detail(`Entry '${id}' is not in '${name}'.`, 404);
      const install = state.installs[id];
      if (!install || install.state === "not_installed") {
        return detail(`'${seed.name}' is not installed.`, 409);
      }
      state.installs[id] = { ...installFrom(stored), enabled: install.enabled };
      persist();
      return HttpResponse.json(toEntry(seed, stored));
    },
  ),

  http.patch(
    "*/api/marketplaces/:name/entries/:entryId",
    async ({ params, request }) => {
      const name = getParam(params.name);
      const id = getParam(params.entryId);
      const stored = findMarketplace(name);
      if (!stored) return marketplaceNotFound(name);
      const seed = stored.entries.find(
        (e) => entryId(e.kind, e.name, name) === id,
      );
      if (!seed) return detail(`Entry '${id}' is not in '${name}'.`, 404);
      const install = state.installs[id];
      if (!install || install.state === "not_installed") {
        return detail(`'${seed.name}' is not installed.`, 409);
      }
      const body = (await request.json()) as { enabled?: boolean };
      state.installs[id] = { ...install, enabled: Boolean(body.enabled) };
      persist();
      return HttpResponse.json(toEntry(seed, stored));
    },
  ),

  http.delete("*/api/marketplaces/:name/entries/:entryId", ({ params }) => {
    const name = getParam(params.name);
    const id = getParam(params.entryId);
    const stored = findMarketplace(name);
    if (!stored) return marketplaceNotFound(name);
    const seed = stored.entries.find(
      (e) => entryId(e.kind, e.name, name) === id,
    );
    if (!seed) return detail(`Entry '${id}' is not in '${name}'.`, 404);
    delete state.installs[id];
    persist();
    return HttpResponse.json(toEntry(seed, stored));
  }),
];

/** Exposed for tests: every kind the default marketplace has at least one of. */
export function defaultMarketplaceKinds(): CatalogKind[] {
  const present = new Set(defaultMarketplaceEntries().map((e) => e.kind));
  return CATALOG_KINDS.filter((kind) => present.has(kind));
}
