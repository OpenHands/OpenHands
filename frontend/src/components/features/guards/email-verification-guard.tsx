import React from "react";
import { useLocation, useNavigate } from "react-router";
import { useSettings } from "#/hooks/query/use-settings";

/**
 * A component that restricts access to routes based on email verification status.
 * If EMAIL_VERIFIED is false, only allows access to the /settings/user page.
 */
export function EmailVerificationGuard({
  children,
}: {
  children: React.ReactNode;
}) {
  const { data: settings, isLoading } = useSettings();
  const navigate = useNavigate();
  const { pathname } = useLocation();

  React.useEffect(() => {
    // If settings are still loading, don't do anything yet
    if (isLoading) return;

    // If EMAIL_VERIFIED is explicitly false (not undefined or null)
    if (settings?.email_verified === false) {
      // Allow access to /settings/user but redirect from any other page
      if (pathname !== "/settings/user") {
        navigate("/settings/user", { replace: true });
      }
    }
  }, [settings?.email_verified, pathname, navigate, isLoading]);

  return children;
}
