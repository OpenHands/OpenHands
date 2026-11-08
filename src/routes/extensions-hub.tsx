import type { RouteTitleHandle } from "./title-handle";
import { I18nKey } from "#/i18n/declaration";
import { Navigate } from "react-router";
import {
  SETTINGS_COMPACT_MAX_WIDTH,
  useBreakpoint,
} from "#/hooks/use-breakpoint";
import { ExtensionsMobileHub } from "#/components/features/skills/extensions-mobile-hub";

export default function ExtensionsHub() {
  const isMobile = useBreakpoint(SETTINGS_COMPACT_MAX_WIDTH);

  if (isMobile) {
    return <ExtensionsMobileHub />;
  }

  return <Navigate to="/mcp" replace />;
}

// @spec AWI-001 — Localized section identity
export const handle = {
  titleKey: I18nKey.NAV$CUSTOMIZE,
} satisfies RouteTitleHandle;
