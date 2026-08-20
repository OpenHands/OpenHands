import { I18nKey } from "#/i18n/declaration";

const SETTINGS_PREFIX = "/settings";
const CUSTOMIZE_HUB = "/customize";
const EXTENSIONS_DETAIL_PATHS = ["/skills", "/mcp", "/plugins"] as const;

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
    EXTENSIONS_DETAIL_PATHS.some(
      (path) => pathname === path || pathname.startsWith(`${path}/`),
    )
  );
}
