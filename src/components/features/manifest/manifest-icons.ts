import {
  Activity,
  Bot,
  CircleAlert,
  LayoutDashboard,
  Sparkles,
  Timer,
  type LucideIcon,
} from "lucide-react";
import type { InterfaceIconSlug } from "#/manifests/types";

/**
 * The artwork behind each icon slug admission accepts. `satisfies` keeps this
 * map and the validator's closed set in lockstep: a slug without artwork, or
 * artwork without a slug, fails to compile.
 */
export const MANIFEST_ICON_BY_SLUG = {
  "layout-dashboard": LayoutDashboard,
  sparkles: Sparkles,
  bot: Bot,
  "circle-alert": CircleAlert,
  activity: Activity,
  timer: Timer,
} satisfies Record<InterfaceIconSlug, LucideIcon>;
