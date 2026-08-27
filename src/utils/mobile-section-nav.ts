import { I18nKey } from "#/i18n/declaration";

const SETTINGS_PREFIX = "/settings";
const CUSTOMIZE_HUB = "/customize";
const EXTENSIONS_DETAIL_PATHS = ["/skills", "/mcp", "/plugins"] as const;
// Exact match only: /extensions/:name/* routes are extension pages, which are
// rail-level destinations rather than Customize details.
const CANVAS_EXTENSIONS_MANAGEMENT_PATH = "/extensions";

export type MobileTopBarMode = "menu" | "back";

export interface MobileTopBarState {
  mode: MobileTopBarMode;
  backTo?: string;
  backLabelKey?: I18nKey;
}

export function getMobileTopBarState(pathname: string): MobileTopBarState {
  if (pathname === SETTINGS_PREFIX) {
    return { mode: "menu" };
  }

  if (
    pathname.startsWith(`${SETTINGS_PREFIX}/`) &&
    pathname.length > SETTINGS_PREFIX.length
  ) {
    return {
      mode: "back",
      backTo: SETTINGS_PREFIX,
      backLabelKey: I18nKey.SETTINGS$TITLE,
    };
  }

  if (pathname === CUSTOMIZE_HUB) {
    return { mode: "menu" };
  }

  if (
    pathname === CANVAS_EXTENSIONS_MANAGEMENT_PATH ||
    EXTENSIONS_DETAIL_PATHS.some(
      (path) => pathname === path || pathname.startsWith(`${path}/`),
    )
  ) {
    return {
      mode: "back",
      backTo: CUSTOMIZE_HUB,
      backLabelKey: I18nKey.NAV$CUSTOMIZE,
    };
  }

  return { mode: "menu" };
}

export function isExtensionsSectionPath(pathname: string): boolean {
  return (
    pathname === CUSTOMIZE_HUB ||
    pathname === CANVAS_EXTENSIONS_MANAGEMENT_PATH ||
    EXTENSIONS_DETAIL_PATHS.some(
      (path) => pathname === path || pathname.startsWith(`${path}/`),
    )
  );
}
