import { Navigate } from "react-router";
import { useBreakpoint } from "#/hooks/use-breakpoint";
import { ExtensionsMobileHub } from "#/components/features/skills/extensions-mobile-hub";

export default function ExtensionsHub() {
  const isMobile = useBreakpoint(768);

  if (isMobile) {
    return <ExtensionsMobileHub />;
  }

  return <Navigate to="/skills" replace />;
}
