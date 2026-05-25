import { openHands } from "../open-hands-axios";

/**
 * Response shape of ``GET /auth/databricks/status``.
 *
 * Mirrors ``openhands/app_server/auth/databricks_routes.py::u2m_status``.
 * Never carries the token itself — only whether the current browser session
 * already has one, and (when authenticated) the workspace host so the UI can
 * show "Signed in to adb-xxx.cloud.databricks.com".
 */
export interface DatabricksAuthStatus {
  /**
   * ``true`` when the deployment has ``DATABRICKS_HOST`` +
   * ``DATABRICKS_U2M_CLIENT_ID`` set. When ``false`` the Sign-in button
   * should be hidden — there is no OAuth app to redirect to.
   */
  configured: boolean;
  /** ``true`` when the current session holds a valid U2M access token. */
  authenticated: boolean;
  /**
   * The workspace URL (only when ``authenticated`` is true). Safe to render
   * in the UI — it's the public workspace URL, not a credential.
   */
  host: string | null;
}

/**
 * Thin client for the Databricks U2M OAuth endpoints exposed at ``/auth/databricks/*``.
 * These routes are mounted at the app root (not under ``/api/v1``) because
 * the OAuth callback URL is registered with Databricks and must not change.
 */
class DatabricksAuthService {
  static readonly INITIATE_URL = "/auth/databricks/initiate";

  /** Read the current U2M OAuth status for the browser session. */
  static async status(): Promise<DatabricksAuthStatus> {
    const { data } = await openHands.get<DatabricksAuthStatus>(
      "/auth/databricks/status",
    );
    return data;
  }

  /**
   * Store OAuth app credentials in the server-side session so that the
   * subsequent browser redirect to ``/initiate`` can use them.
   *
   * Call this BEFORE redirecting to ``INITIATE_URL``.  The server stores
   * ``client_id`` (and optional ``client_secret`` for confidential apps) in
   * the signed session cookie — they are never sent back to the browser.
   *
   * Returns the URL to redirect to (always ``/auth/databricks/initiate``).
   */
  static async prepare(
    clientId: string,
    host: string,
    clientSecret?: string | null,
    redirectUri?: string | null,
  ): Promise<string> {
    const { data } = await openHands.post<{ redirect_url: string }>(
      "/auth/databricks/prepare",
      {
        client_id: clientId,
        host,
        client_secret: clientSecret || null,
        redirect_uri: redirectUri || null,
      },
    );
    return data.redirect_url;
  }

  /**
   * Clear the U2M tokens from the session. Does not revoke at Databricks —
   * a subsequent Sign-in will re-use the browser's SSO if still active.
   */
  static async logout(): Promise<void> {
    await openHands.post("/auth/databricks/logout");
  }
}

export default DatabricksAuthService;
