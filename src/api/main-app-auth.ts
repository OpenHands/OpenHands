import axios from "axios";
import { getLockedCloudAuthMode } from "./agent-server-config";

export const MAIN_APP_AUTHENTICATE_PATH = "/api/authenticate";
export const MAIN_APP_LOGIN_PATH = "/login";
export const MAIN_APP_LOGIN_REDIRECT_PARAM = "returnTo";

const LOCAL_BROWSER_HOSTNAMES = new Set(["localhost", "127.0.0.1", "::1"]);

function getCurrentPathForRedirect(): string {
  if (typeof window === "undefined") return "/";
  return `${window.location.pathname}${window.location.search}${window.location.hash}`;
}

export function isLocalBrowserHost(): boolean {
  if (typeof window === "undefined") return false;
  return LOCAL_BROWSER_HOSTNAMES.has(window.location.hostname);
}

export function shouldUseMainAppCookieAuth(): boolean {
  return getLockedCloudAuthMode() === "cookie" && !isLocalBrowserHost();
}

export async function authenticateWithMainAppCookie(): Promise<boolean> {
  try {
    await axios.post(MAIN_APP_AUTHENTICATE_PATH, null, {
      withCredentials: true,
    });
    return true;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response?.status === 401) {
      return false;
    }
    throw error;
  }
}

export function buildMainAppLoginRedirect(
  redirectPath = getCurrentPathForRedirect(),
): string {
  const params = new URLSearchParams({
    [MAIN_APP_LOGIN_REDIRECT_PARAM]: redirectPath,
  });
  return `${MAIN_APP_LOGIN_PATH}?${params.toString()}`;
}

export function redirectToMainAppLogin(): void {
  if (typeof window === "undefined") return;
  window.location.assign(buildMainAppLoginRedirect());
}
