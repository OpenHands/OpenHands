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
