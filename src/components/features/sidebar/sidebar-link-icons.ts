import { BookOpen, Cloud, ExternalLink, LifeBuoy } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { SidebarLinkIconSlug } from "#/config/sidebar-links";

/**
 * The artwork behind each sidebar link icon slug. `satisfies` keeps this map
 * and the config validator's closed set in lockstep: a slug without artwork,
 * or artwork without a slug, fails to compile.
 */
export const SIDEBAR_LINK_ICON_BY_SLUG = {
  "external-link": ExternalLink,
  cloud: Cloud,
  "book-open": BookOpen,
  "life-buoy": LifeBuoy,
} satisfies Record<SidebarLinkIconSlug, LucideIcon>;
