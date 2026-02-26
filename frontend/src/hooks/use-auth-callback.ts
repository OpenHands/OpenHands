import { useEffect } from "react";
import { useLocation, useNavigate } from "react-router";
import { useIsAuthed } from "./query/use-is-authed";
import { LoginMethod, setLoginMethod } from "#/utils/local-storage";
import { useConfig } from "./query/use-config";

/**
 * Hook to handle authentication callback and set login method after successful authentication
 */
export const useAuthCallback = () => {
  const location = useLocation();
  const { data: isAuthed, isLoading: isAuthLoading } = useIsAuthed();
  const { data: config } = useConfig();
  const navigate = useNavigate();

  useEffect(() => {
    // Only run in SAAS mode
    if (config?.app_mode !== "saas") {
      return;
    }

    // Wait for auth to load
    if (isAuthLoading) {
      return;
    }

    // Only process callback if authentication was successful
    if (!isAuthed) {
      return;
    }

    // Check if we have a login_method query parameter
    const searchParams = new URLSearchParams(location.search);
    const loginMethod = searchParams.get("login_method");
    const returnTo = searchParams.get("returnTo");

    // Set the login method if it's valid
    if (Object.values(LoginMethod).includes(loginMethod as LoginMethod)) {
      setLoginMethod(loginMethod as LoginMethod);

      // Clean up the URL by removing auth-related parameters
      searchParams.delete("login_method");
      searchParams.delete("returnTo");

      // Determine where to navigate after authentication
      let destination = "/";
      if (returnTo && returnTo !== "/login") {
        destination = returnTo;
      } else if (location.pathname !== "/login" && location.pathname !== "/") {
        destination = location.pathname;
      }

      const remainingParams = searchParams.toString();
      const finalUrl = remainingParams
        ? `${destination}?${remainingParams}`
        : destination;

      navigate(finalUrl, { replace: true });
    }
  }, [
    isAuthed,
    isAuthLoading,
    location.search,
    location.pathname,
    config?.app_mode,
    navigate,
  ]);
};
