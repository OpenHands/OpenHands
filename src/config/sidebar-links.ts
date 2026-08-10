/**
 * Operator-configured external links appended to the sidebar navigation
 * (e.g. an OpenHands Enterprise link), added without a Canvas code change.
 *
 * Two sources are consulted, in order — mirroring getBakedSessionApiKey() /
 * getLockedCloudHost() in `agent-server-config.ts`:
 *   1. `VITE_SIDEBAR_LINKS` — baked into the bundle at build time (dev server).
 *   2. `window.__AGENT_CANVAS_SIDEBAR_LINKS__` — injected into index.html at
 *      serve time by `scripts/static-server.mjs --sidebar-links <json>`, fed
 *      from the `AGENT_CANVAS_SIDEBAR_LINKS` env var in
 *      `docker/entrypoint.sh`. This is what lets an operator add a link to a
 *      published build without a rebuild.
 *
 * Each entry is data, not instructions: a malformed or unsafe entry (missing
 * fields, non-http(s) URL, unknown icon slug, duplicate id) is dropped rather
 * than partially rendered.
 */

export const SIDEBAR_LINK_ICON_SLUGS = [
  "external-link",
  "cloud",
  "book-open",
  "life-buoy",
] as const;

export type SidebarLinkIconSlug = (typeof SIDEBAR_LINK_ICON_SLUGS)[number];

export interface SidebarLink {
  id: string;
  label: string;
  url: string;
  icon: SidebarLinkIconSlug;
}

const ID_PATTERN = /^[a-z0-9]+(-[a-z0-9]+)*$/;

function isSafeUrl(value: string): boolean {
  try {
    const parsed = new URL(value);
    return parsed.protocol === "https:" || parsed.protocol === "http:";
  } catch {
    return false;
  }
}

function isIconSlug(value: unknown): value is SidebarLinkIconSlug {
  return (
    typeof value === "string" &&
    (SIDEBAR_LINK_ICON_SLUGS as readonly string[]).includes(value)
  );
}

function validateEntry(candidate: unknown, index: number): SidebarLink | null {
  if (typeof candidate !== "object" || candidate === null) {
    console.warn(`Rejected sidebar link at index ${index}: not an object`);
    return null;
  }

  const { id, label, url, icon } = candidate as Record<string, unknown>;

  const safeId = typeof id === "string" && ID_PATTERN.test(id) ? id : null;
  if (!safeId) {
    console.warn(
      `Rejected sidebar link at index ${index}: "id" must be a kebab-case string`,
    );
    return null;
  }

  if (typeof label !== "string" || !label.trim()) {
    console.warn(`Rejected sidebar link "${safeId}": "label" is required`);
    return null;
  }

  if (typeof url !== "string" || !isSafeUrl(url)) {
    console.warn(
      `Rejected sidebar link "${safeId}": "url" must be an http(s) URL`,
    );
    return null;
  }

  return {
    id: safeId,
    label,
    url,
    icon: isIconSlug(icon) ? icon : "external-link",
  };
}

function parseSidebarLinks(raw: string | null | undefined): SidebarLink[] {
  if (!raw) return [];

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    console.warn("Rejected sidebar links config: invalid JSON");
    return [];
  }

  if (!Array.isArray(parsed)) {
    console.warn("Rejected sidebar links config: expected a JSON array");
    return [];
  }

  const seen = new Set<string>();
  const links: SidebarLink[] = [];

  parsed.forEach((candidate, index) => {
    const link = validateEntry(candidate, index);
    if (!link) return;

    if (seen.has(link.id)) {
      console.warn(`Rejected sidebar link: id "${link.id}" is already used`);
      return;
    }

    seen.add(link.id);
    links.push(link);
  });

  return links;
}

function getRawSidebarLinksConfig(): string | null {
  const envValue = import.meta.env.VITE_SIDEBAR_LINKS;
  if (envValue) return envValue;

  if (typeof window !== "undefined") {
    const injected = (window as unknown as Record<string, unknown>)
      .__AGENT_CANVAS_SIDEBAR_LINKS__;
    if (typeof injected === "string") return injected;
  }

  return null;
}

let cached: SidebarLink[] | null = null;

/** Operator-configured sidebar links, parsed and validated once per session. */
export function getConfiguredSidebarLinks(): SidebarLink[] {
  if (cached === null) {
    cached = parseSidebarLinks(getRawSidebarLinksConfig());
  }
  return cached;
}

/** Test-only: clears the memoized result so a test can re-seed config. */
export function resetConfiguredSidebarLinksCache(): void {
  cached = null;
}
