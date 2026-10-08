/**
 * DigitalOcean OAuth sign-in for the MARS backend.
 *
 * Uses the **implicit** grant (`response_type=token`), which DigitalOcean
 * documents as the flow for "client-side applications such as mobile or
 * desktop clients where the client secret should not be stored on the user's
 * device". DigitalOcean documents no PKCE support, so the authorization-code
 * grant would require shipping a `client_secret` inside a distributed desktop
 * binary — which is not a secret at all. The tradeoff is that implicit issues
 * no refresh token, so the 30-day expiry is handled by re-authenticating.
 */

import { createServer } from "node:http";
import { randomUUID } from "node:crypto";

const AUTHORIZE_URL = "https://cloud.digitalocean.com/v1/oauth/authorize";
const REVOKE_URL = "https://cloud.digitalocean.com/v1/oauth/revoke";

/**
 * DigitalOcean matches `redirect_uri` against the value registered with the
 * OAuth application, so the loopback port cannot be OS-assigned — it has to be
 * a fixed, registered port. (rclone pins 53682 for the same reason.)
 */
export const OAUTH_REDIRECT_PORT = 53682;
export const OAUTH_REDIRECT_URI = `http://127.0.0.1:${OAUTH_REDIRECT_PORT}/callback`;

/** Permissions the edge enforces on `/v2/agents/*` for MARS. */
export const OAUTH_SCOPES = [
  "agent_harness_session:read",
  "agent_harness_session:operate",
].join(" ");

const CALLBACK_PATH = "/callback";
const RELAY_PATH = "/callback-token";
const SIGN_IN_TIMEOUT_MS = 5 * 60 * 1000;

/**
 * Served at the redirect URI.
 *
 * The implicit grant returns the token in the URL *fragment*, and browsers
 * never transmit fragments to the server — so the loopback request that lands
 * here carries no token at all. This page exists purely to read
 * `location.hash` in the browser and POST it back on a second request.
 * Without it the flow appears to hang forever.
 */
const RELAY_PAGE = `<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Signing in…</title>
<style>
  body { font-family: ui-sans-serif, system-ui, sans-serif; background: #0b0e14;
         color: #e6e6e6; display: grid; place-items: center; height: 100vh; margin: 0; }
  .card { text-align: center; max-width: 26rem; padding: 2rem; }
  h1 { font-size: 1.1rem; font-weight: 600; }
  p { color: #9aa0aa; font-size: .9rem; line-height: 1.5; }
</style>
</head>
<body>
  <div class="card">
    <h1 id="title">Finishing sign-in…</h1>
    <p id="detail">You can close this tab in a moment.</p>
  </div>
  <script>
    (function () {
      var fragment = window.location.hash.replace(/^#/, "");
      fetch(${JSON.stringify(RELAY_PATH)}, {
        method: "POST",
        headers: { "Content-Type": "text/plain" },
        body: fragment,
      }).then(function () {
        document.getElementById("title").textContent = "Signed in to DigitalOcean";
        document.getElementById("detail").textContent = "You can close this tab and return to OpenHands.";
      }).catch(function () {
        document.getElementById("title").textContent = "Sign-in could not be completed";
        document.getElementById("detail").textContent = "Return to OpenHands and try again, or use a personal access token.";
      });
    })();
  </script>
</body>
</html>`;

/**
 * @param {object} options
 * @param {string} options.clientId
 * @param {string} options.state
 * @param {string} [options.redirectUri]
 * @param {string} [options.scope]
 */
export function buildAuthorizeUrl({ clientId, state, redirectUri, scope }) {
  const url = new URL(AUTHORIZE_URL);
  url.searchParams.set("client_id", clientId);
  url.searchParams.set("redirect_uri", redirectUri ?? OAUTH_REDIRECT_URI);
  url.searchParams.set("response_type", "token");
  url.searchParams.set("scope", scope ?? OAUTH_SCOPES);
  url.searchParams.set("state", state);
  url.searchParams.set("prompt", "select_account");
  return url.toString();
}

/**
 * Parse the relayed fragment into a grant. Also handles the error form
 * DigitalOcean redirects with when the user declines or is not signed in.
 *
 * @param {string | null | undefined} fragment
 * @param {string} expectedState
 */
export function parseImplicitGrant(fragment, expectedState) {
  const params = new URLSearchParams(fragment ?? "");

  const error = params.get("error");
  if (error) {
    throw new Error(params.get("error_description") || error);
  }

  const accessToken = params.get("access_token");
  if (!accessToken) {
    throw new Error("DigitalOcean did not return an access token.");
  }

  // Guards against a forged callback: a third party that can reach the
  // loopback port cannot know the state we generated for this attempt.
  if (params.get("state") !== expectedState) {
    throw new Error("Sign-in state did not match; aborting for safety.");
  }

  const expiresIn = Number.parseInt(params.get("expires_in") ?? "", 10);
  return {
    accessToken,
    tokenType: params.get("token_type") ?? "bearer",
    expiresAt: Number.isFinite(expiresIn)
      ? new Date(Date.now() + expiresIn * 1000).toISOString()
      : null,
  };
}

function readBody(req) {
  return new Promise((resolve) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", () => resolve(body));
  });
}

/**
 * Run the full sign-in: open the system browser, wait for the relayed grant.
 *
 * The system browser is used rather than an in-app `BrowserWindow` so the user
 * authorizes in the session they are already signed into and can see the real
 * URL they are authenticating against.
 *
 * @param {object} options
 * @param {string | null | undefined} options.clientId
 * @param {(url: string) => Promise<void>} options.openExternal
 * @param {number} [options.timeoutMs]
 */
export async function signInWithDigitalOcean({
  clientId,
  openExternal,
  timeoutMs = SIGN_IN_TIMEOUT_MS,
}) {
  if (!clientId) {
    throw new Error(
      "No DigitalOcean OAuth client is configured in this build. Use a personal access token instead.",
    );
  }

  const state = randomUUID();
  const server = createServer();

  const grant = new Promise((resolve, reject) => {
    server.on("request", async (req, res) => {
      const { pathname } = new URL(req.url, OAUTH_REDIRECT_URI);

      if (req.method === "GET" && pathname === CALLBACK_PATH) {
        res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
        res.end(RELAY_PAGE);
        return;
      }

      if (req.method === "POST" && pathname === RELAY_PATH) {
        const fragment = await readBody(req);
        try {
          const parsed = parseImplicitGrant(fragment, state);
          res.writeHead(204).end();
          resolve(parsed);
        } catch (error) {
          res.writeHead(400).end();
          reject(error);
        }
        return;
      }

      res.writeHead(404).end();
    });
  });

  // One handler for listen and runtime errors; racing it everywhere keeps the
  // EADDRINUSE explanation from being shadowed by the raw listen error.
  const failed = new Promise((_, reject) => {
    server.on("error", (error) => {
      reject(
        error.code === "EADDRINUSE"
          ? new Error(
              `Port ${OAUTH_REDIRECT_PORT} is already in use, so DigitalOcean cannot redirect back. Close whatever is using it, or sign in with a personal access token instead.`,
            )
          : error,
      );
    });
  });

  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(
      () => reject(new Error("Timed out waiting for DigitalOcean sign-in.")),
      timeoutMs,
    );
    timer.unref?.();
  });

  try {
    await Promise.race([
      new Promise((resolve) =>
        server.listen(OAUTH_REDIRECT_PORT, "127.0.0.1", resolve),
      ),
      failed,
    ]);

    await openExternal(buildAuthorizeUrl({ clientId, state }));
    return await Promise.race([grant, failed, timeout]);
  } finally {
    clearTimeout(timer);
    server.close();
  }
}

/**
 * Invalidate an access token server-side. Best-effort: a failed revoke must
 * not prevent the local credential from being forgotten.
 */
export async function revokeToken(token, fetchImpl = globalThis.fetch) {
  try {
    await fetchImpl(REVOKE_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({ token }).toString(),
    });
  } catch {
    // Network failure on sign-out; the local record is still removed.
  }
}
