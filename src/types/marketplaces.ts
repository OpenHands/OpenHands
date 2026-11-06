/**
 * Marketplace sources and their unified catalog.
 *
 * A marketplace is a repository laid out like `OpenHands/extensions`. Its
 * catalog lists every kind of Customize resource it ships, so one page can
 * browse, install and update all of them.
 */

/** Who registered the marketplace. Narrower scopes add, they never override. */
export type MarketplaceScope = "instance" | "org" | "user";

export type CatalogKind = "mcp" | "skill" | "plugin" | "app" | "automation";

export const CATALOG_KINDS: readonly CatalogKind[] = [
  "mcp",
  "skill",
  "plugin",
  "app",
  "automation",
];

export interface MarketplaceSource {
  /** Stable identifier; entries are addressed as `name@marketplace`. */
  name: string;
  /** `github:owner/repo`, a git URL or a local path. */
  source: string;
  ref: string | null;
  repo_path: string | null;
  scope: MarketplaceScope;
  /** Content that runs code may only be installed from a trusted source. */
  trusted: boolean;
  /** Name of the secret holding the access token for a private repository. */
  token_secret: string | null;
  /** The instance default cannot be removed. */
  is_default: boolean;
  resolved_ref: string | null;
  last_refreshed_at: string | null;
  counts: Record<CatalogKind, number>;
  error_count: number;
}

export type InstallState = "not_installed" | "installed" | "update_available";

export interface CatalogEntryInstall {
  state: InstallState;
  enabled: boolean;
  /** Commit the installed copy came from. */
  installed_ref: string | null;
  /** `track` follows the marketplace ref; `pin` stays on `installed_ref`. */
  policy: "track" | "pin";
}

export interface CatalogEntry {
  /** `${kind}:${name}@${marketplace}` */
  id: string;
  kind: CatalogKind;
  name: string;
  marketplace: string;
  description: string;
  tags: string[];
  version: string | null;
  /** Whether installing it brings executable content (hooks, bundles, stdio). */
  runs_code: boolean;
  /** Other marketplaces that ship an entry of the same kind and name. */
  also_in: string[];
  install: CatalogEntryInstall;
  /** Automation templates: integration ids the template needs connected. */
  requires: string[];
  /** MCP servers: how the server is reached. */
  transport: "stdio" | "http" | null;
}

export interface CatalogError {
  /** Entry name as it appears in the marketplace manifest. */
  entry: string;
  kind: CatalogKind | null;
  message: string;
}

export interface MarketplaceCatalog {
  marketplace: MarketplaceSource;
  entries: CatalogEntry[];
  errors: CatalogError[];
}

export interface AddMarketplaceRequest {
  name?: string;
  source: string;
  ref?: string | null;
  trusted: boolean;
  token_secret?: string | null;
}

export interface UpdateMarketplaceRequest {
  trusted?: boolean;
}
