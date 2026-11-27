import type {
  MarketplacePlugin,
  LocalPlugin,
  PluginBundledSkill,
} from "#/api/plugins-service";
import type { InstalledPluginInfo } from "#/api/plugins-management-service";

export type PluginStatusFilter = "all" | "installed" | "available" | "local";

/**
 * A single row in the plugins management list, reconciling the dynamic
 * marketplace catalog with the locally-installed plugins.
 */
export interface PluginViewModel {
  name: string;
  description: string | null;
  source: string | null;
  ref: string | null;
  repoPath: string | null;
  /** Installed on the local agent-server. */
  installed: boolean;
  /** Enabled (only meaningful when installed); enabled plugins auto-load. */
  enabled: boolean;
  /** Installed version, when known. */
  version: string | null;
  /** Present in the marketplace catalog. */
  inCatalog: boolean;
  /**
   * Discovered from a local ambient directory (e.g. `~/.agents/plugins`).
   * Read-only: it auto-loads into conversations and is not installed/managed.
   */
  isLocal: boolean;
  /**
   * Plugin contents (`skills` bundled in the plugin, `files` relative to
   * `path`). The three fields travel as a unit from one server response so the
   * file viewer never joins a base path with another copy's file list; null
   * when the server has no local copy or predates the contents fields.
   */
  path: string | null;
  skills: PluginBundledSkill[] | null;
  files: string[] | null;
}

/**
 * Merge the marketplace catalog with the installed plugins into one
 * de-duplicated list keyed by plugin name. The installed list is the source of
 * truth for install/enable state and coordinates; the catalog supplies
 * description/coordinates as a fallback. Installed plugins sort first, then
 * alphabetically.
 */
export function buildPluginsViewModel(
  marketplace: MarketplacePlugin[] | undefined,
  installed: InstalledPluginInfo[] | undefined,
  local: LocalPlugin[] | undefined = undefined,
): PluginViewModel[] {
  const byName = new Map<string, PluginViewModel>();

  for (const entry of marketplace ?? []) {
    byName.set(entry.name, {
      name: entry.name,
      description: entry.description,
      source: entry.source,
      ref: entry.ref ?? null,
      repoPath: entry.repo_path ?? null,
      installed: entry.installed,
      enabled: false,
      version: null,
      inCatalog: true,
      isLocal: false,
      path: entry.path ?? null,
      skills: entry.skills ?? null,
      files: entry.files ?? null,
    });
  }

  for (const entry of installed ?? []) {
    const existing = byName.get(entry.name);
    // Contents move as a unit: the installed copy is authoritative when it
    // carries contents; otherwise (older agent-server) keep the catalog's.
    const hasContents = entry.skills != null || entry.files != null;
    byName.set(entry.name, {
      name: entry.name,
      description: existing?.description ?? entry.description ?? null,
      source: entry.source ?? existing?.source ?? null,
      ref: entry.resolved_ref ?? existing?.ref ?? null,
      repoPath: entry.repo_path ?? existing?.repoPath ?? null,
      installed: true,
      enabled: entry.enabled,
      version: entry.version ?? null,
      inCatalog: existing?.inCatalog ?? false,
      isLocal: false,
      path: hasContents ? entry.install_path : (existing?.path ?? null),
      skills: hasContents ? (entry.skills ?? null) : (existing?.skills ?? null),
      files: hasContents ? (entry.files ?? null) : (existing?.files ?? null),
    });
  }

  // Ambient local plugins are additive and read-only. A local plugin whose name
  // is already installed or in the catalog is skipped — that managed entry is
  // authoritative (the agent-server also folds enabled installed plugins into
  // this list), so it must not spawn a duplicate read-only "Local" card.
  for (const entry of local ?? []) {
    if (byName.has(entry.name)) continue;
    byName.set(entry.name, {
      name: entry.name,
      description: entry.description || null,
      source: null,
      ref: null,
      repoPath: null,
      installed: false,
      enabled: false,
      version: entry.version || null,
      inCatalog: false,
      isLocal: true,
      path: entry.path ?? null,
      skills: entry.skills ?? null,
      files: entry.files ?? null,
    });
  }

  // Sort installed first, then local, then available; alphabetical within each.
  const rank = (plugin: PluginViewModel): number =>
    plugin.installed ? 0 : plugin.isLocal ? 1 : 2;

  return Array.from(byName.values()).sort((a, b) => {
    const rankDelta = rank(a) - rank(b);
    if (rankDelta !== 0) return rankDelta;
    return a.name.localeCompare(b.name);
  });
}

/** True when the plugin matches a free-text search query (empty query matches). */
export function matchesPluginSearch(
  plugin: PluginViewModel,
  query: string,
): boolean {
  const trimmed = query.trim().toLowerCase();
  if (!trimmed) return true;
  const haystacks = [
    plugin.name,
    plugin.description ?? "",
    plugin.source ?? "",
    plugin.repoPath ?? "",
    plugin.ref ?? "",
  ];
  return haystacks.some((value) => value.toLowerCase().includes(trimmed));
}

/** True when the plugin matches the install-state filter. */
export function matchesPluginStatus(
  plugin: PluginViewModel,
  filter: PluginStatusFilter,
): boolean {
  if (filter === "installed") return plugin.installed;
  // "Available" means installable from the catalog; local plugins are neither
  // installed nor installable, so they are excluded here and grouped under
  // "local" instead.
  if (filter === "available") return !plugin.installed && !plugin.isLocal;
  if (filter === "local") return plugin.isLocal;
  return true;
}
