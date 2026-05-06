import { openHands } from "../open-hands-axios";
import { AuthenticateResponse, GitHubAccessTokenResponse } from "./auth.types";
import { GetConfigResponse } from "../option-service/option.types";

/**
 * Authentication service for handling all authentication-related API calls
 */
class AuthService {
  /**
   * Authenticate with Better Auth session or legacy flow
   * @param appMode The application mode (saas, oss, or b1)
   * @returns Response with authentication status and user info if successful
   */
  static async authenticate(
    appMode: GetConfigResponse["APP_MODE"],
  ): Promise<boolean> {
    if (appMode === "oss") return true;

    // Just make the request, if it succeeds (no exception thrown), return true
    await openHands.post<AuthenticateResponse>("/api/authenticate");
    return true;
  }

  /**
   * Sign in with email and password via Better Auth
   * @param email User email
   * @param password User password
   */
  static async signIn(email: string, password: string): Promise<void> {
    await openHands.post("/api/auth/sign-in", { email, password });
  }

  /**
   * Sign up with email, password, and name via Better Auth
   * @param email User email
   * @param password User password
   * @param name User display name
   */
  static async signUp(
    email: string,
    password: string,
    name: string,
  ): Promise<void> {
    await openHands.post("/api/auth/sign-up", { email, password, name });
  }

  /**
   * Get OAuth redirect URL for a given provider
   * @param provider OAuth provider name (e.g., "github", "google")
   * @param callbackURL URL to redirect back to after OAuth
   * @returns The OAuth redirect URL
   */
  static async getOAuthUrl(
    provider: string,
    callbackURL: string = "/",
  ): Promise<string> {
    const { data } = await openHands.post<{ url: string }>(
      "/api/auth/sign-in/social",
      { provider, callbackURL },
    );
    return data.url;
  }

  /**
   * Get available OAuth providers from the server
   * @returns Array of provider names
   */
  static async getProviders(): Promise<string[]> {
    const { data } = await openHands.get<{ providers: string[] }>(
      "/api/auth/providers",
    );
    return data.providers || [];
  }

  /**
   * Get GitHub access token from Keycloak callback
   * @param code Code provided by GitHub
   * @returns GitHub access token
   */
  static async getGitHubAccessToken(
    code: string,
  ): Promise<GitHubAccessTokenResponse> {
    const { data } = await openHands.post<GitHubAccessTokenResponse>(
      "/api/keycloak/callback",
      {
        code,
      },
    );
    return data;
  }

  /**
   * Redeem a single-use handoff code for a session cookie at our origin.
   * Throws on non-2xx so callers can decide whether to fall through to
   * the existing sign-in flow.
   */
  static async redeemHandoffCode(code: string): Promise<void> {
    await openHands.post("/api/auth/handoff/redeem", { code });
  }

  /**
   * Logout user from the application
   * @param appMode The application mode (saas, oss, or b1)
   */
  static async logout(appMode: GetConfigResponse["APP_MODE"]): Promise<void> {
    if (appMode === "saas" || appMode === "b1") {
      await openHands.post("/api/logout");
      return;
    }
    await openHands.post("/api/unset-provider-tokens");
  }
}

export default AuthService;
