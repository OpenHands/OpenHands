/**
 * DigitalOcean OAuth sign-in for the MARS backend.
 *
 * Uses the authorization-code grant with PKCE, which DigitalOcean requires of
 * public clients: a desktop app or a self-hosted Canvas server cannot keep a
 * client secret. The PKCE proof never leaves this process, so a return
 * address that leaks (a pasted URL, a log line) is useless on its own. Access
 * tokens last an hour and come with a refresh token that works exactly once.
 *
 * DigitalOcean matches loopback return addresses (127.0.0.1, [::1],
 * localhost) on any port, so a browser on the Canvas host comes back to the
 * page's own origin, and the desktop app listens on a port of its choosing.
 */

import { createHash, randomBytes, randomUUID } from "node:crypto";
import { createServer } from "node:http";

const AUTHORIZE_URL = "https://cloud.digitalocean.com/v1/oauth/authorize";
const TOKEN_URL = "https://cloud.digitalocean.com/v1/oauth/token";
const REVOKE_URL = "https://cloud.digitalocean.com/v1/oauth/revoke";

/**
 * Permissions required by the agent/session actions Canvas exposes. A grant
 * issued with fewer scopes needs a new sign-in; refreshing cannot add scopes.
 */
export const OAUTH_SCOPES = [
  "agent_harness_session:read",
  "agent_harness_session:create",
  "agent_harness_session:operate",
  "agent_harness_session:manage",
].join(" ");

/** Where a browser comes back to on the Canvas server (scripts/mars-web-bridge.mjs). */
export const OAUTH_CALLBACK_PATH = "/mars/oauth/callback";
/** Where the desktop app's one-off loopback listener takes the return. */
const DESKTOP_CALLBACK_PATH = "/callback";

const SIGN_IN_TIMEOUT_MS = 5 * 60 * 1000;
/** A started sign-in can be finished for this long, and only once. */
const ATTEMPT_TTL_MS = 10 * 60 * 1000;
const TOKEN_TIMEOUT_MS = 15_000;

/** A refusal from DigitalOcean's token endpoint, or a failure to reach it. */
export class OAuthError extends Error {
  /**
   * @param {string} message
   * @param {{ status?: number | null, code?: string | null, cause?: unknown }} [details]
   */
  constructor(message, { status = null, code = null, cause } = {}) {
    super(message, { cause });
    this.name = "OAuthError";
    this.status = status;
    this.code = code;
  }

  /**
   * The grant is gone (used, revoked or expired): only a new sign-in helps.
   * DigitalOcean answers a spent refresh token with `invalid_request`, so
   * any refusal counts, not just `invalid_grant`; only a failure to reach
   * DigitalOcean (no status) is worth retrying.
   */
  get isRevoked() {
    return this.status === 400 || this.status === 401;
  }
}

/** A PKCE pair: the verifier stays here, the challenge goes to DigitalOcean. */
export function createPkce() {
  const verifier = randomBytes(32).toString("base64url");
  const challenge = createHash("sha256").update(verifier).digest("base64url");
  return { verifier, challenge };
}

/**
 * @param {object} options
 * @param {string} options.clientId
 * @param {string} options.redirectUri
 * @param {string} options.state
 * @param {string} options.codeChallenge
 * @param {string} [options.scope]
 */
export function buildAuthorizeUrl({
  clientId,
  redirectUri,
  state,
  codeChallenge,
  scope = OAUTH_SCOPES,
}) {
  const url = new URL(AUTHORIZE_URL);
  url.searchParams.set("client_id", clientId);
  url.searchParams.set("redirect_uri", redirectUri);
  url.searchParams.set("response_type", "code");
  url.searchParams.set("scope", scope);
  url.searchParams.set("state", state);
  url.searchParams.set("code_challenge", codeChallenge);
  url.searchParams.set("code_challenge_method", "S256");
  url.searchParams.set("prompt", "select_account");
  return url.toString();
}

/**
 * What DigitalOcean sent back, from the full return address, the path and
 * query the return route receives, or just the query: a code, or the error
 * it redirects with when the user declines.
 *
 * @param {string | null | undefined} returned
 * @returns {{ code: string | null, state: string | null, error: string | null }}
 */
export function readCallback(returned) {
  const text = String(returned ?? "").trim();
  let params;
  try {
    params = new URL(text).searchParams;
  } catch {
    const query = text.includes("?") ? text.slice(text.indexOf("?") + 1) : text;
    params = new URLSearchParams(query.replace(/^#/, ""));
  }
  const error = params.get("error");
  return {
    code: params.get("code"),
    state: params.get("state"),
    error: error ? params.get("error_description") || error : null,
  };
}

/**
 * The token response as a grant. Its `info` names the account and team, so
 * a sign-in is labelled without a second call.
 */
function toGrant(body) {
  if (!body?.access_token) {
    throw new OAuthError("DigitalOcean did not return an access token.");
  }
  const expiresIn = Number(body.expires_in);
  const info = body.info ?? null;
  return {
    accessToken: body.access_token,
    refreshToken: body.refresh_token ?? null,
    accessExpiresAt:
      Number.isFinite(expiresIn) && expiresIn > 0
        ? new Date(Date.now() + expiresIn * 1000).toISOString()
        : null,
    account: info
      ? {
          uuid: info.team_uuid ?? null,
          name: info.team_name ?? null,
          email: info.email ?? null,
        }
      : null,
  };
}

async function requestToken(params, fetchImpl) {
  let response;
  try {
    response = await fetchImpl(TOKEN_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        Accept: "application/json",
      },
      body: new URLSearchParams(params).toString(),
      signal: AbortSignal.timeout(TOKEN_TIMEOUT_MS),
    });
  } catch (error) {
    throw new OAuthError("Could not reach DigitalOcean to sign in.", {
      cause: error,
    });
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new OAuthError(
      body?.error_description ||
        body?.error ||
        `DigitalOcean refused the sign-in (${response.status}).`,
      { status: response.status, code: body?.error ?? null },
    );
  }
  return toGrant(body);
}

/**
 * Trade the code DigitalOcean sent back, with the PKCE verifier, for tokens.
 *
 * @param {object} options
 * @param {string} options.clientId
 * @param {string} options.code
 * @param {string} options.verifier
 * @param {string} options.redirectUri The same address the sign-in started with
 * @param {typeof fetch} [options.fetchImpl]
 */
export function exchangeCode({
  clientId,
  code,
  verifier,
  redirectUri,
  fetchImpl = globalThis.fetch,
}) {
  return requestToken(
    {
      grant_type: "authorization_code",
      code,
      client_id: clientId,
      redirect_uri: redirectUri,
      code_verifier: verifier,
    },
    fetchImpl,
  );
}

/**
 * A new access token, and a new refresh token: the one passed in is spent
 * whether or not the caller manages to save the result.
 *
 * @param {object} options
 * @param {string} options.clientId
 * @param {string} options.refreshToken
 * @param {typeof fetch} [options.fetchImpl]
 */
export function refreshGrant({
  clientId,
  refreshToken,
  fetchImpl = globalThis.fetch,
}) {
  return requestToken(
    {
      grant_type: "refresh_token",
      refresh_token: refreshToken,
      client_id: clientId,
    },
    fetchImpl,
  );
}

/**
 * Sign-ins in progress. Each keeps its PKCE verifier and is found by the
 * `state` DigitalOcean echoes back. It can be finished once, within ten
 * minutes; a replayed or late return finds nothing.
 *
 * @param {{ ttlMs?: number, now?: () => number }} [options]
 */
export function createSignInAttempts({
  ttlMs = ATTEMPT_TTL_MS,
  now = Date.now,
} = {}) {
  /**
   * @type {Map<string, { id: string, state: string, verifier: string, clientId: string, redirectUri: string, createdAt: number, status: "pending" | "working" | "done" | "failed", message: string | null, cancelled?: boolean, waiters: Set<() => void> }>}
   */
  const attempts = new Map();

  function settle(attempt, status, message = null) {
    attempt.status = status;
    attempt.message = message;
    for (const wake of attempt.waiters) wake();
    attempt.waiters.clear();
  }

  function prune() {
    for (const [id, attempt] of attempts) {
      if (now() - attempt.createdAt <= ttlMs) continue;
      // A return claimed just in time is still being finished; it settles,
      // and is pruned, once its exchange answers.
      if (attempt.status === "working") continue;
      if (attempt.status === "pending") {
        settle(attempt, "failed", "This sign-in expired. Start again.");
      }
      attempts.delete(id);
    }
  }

  function view(attempt) {
    if (!attempt) {
      return {
        status: "failed",
        message: "This sign-in is no longer open. Start again.",
      };
    }
    const pending =
      attempt.status === "pending" || attempt.status === "working";
    return pending
      ? { status: "pending" }
      : { status: attempt.status, message: attempt.message };
  }

  return {
    /**
     * @param {{ clientId: string, redirectUri: string }} options
     * @returns {{ attemptId: string, authorizeUrl: string }}
     */
    begin({ clientId, redirectUri }) {
      prune();
      const { verifier, challenge } = createPkce();
      const attempt = {
        id: randomUUID(),
        state: randomBytes(24).toString("base64url"),
        verifier,
        clientId,
        redirectUri,
        createdAt: now(),
        status: /** @type {const} */ ("pending"),
        message: null,
        waiters: new Set(),
      };
      attempts.set(attempt.id, attempt);
      return {
        attemptId: attempt.id,
        authorizeUrl: buildAuthorizeUrl({
          clientId,
          redirectUri,
          state: attempt.state,
          codeChallenge: challenge,
        }),
      };
    },

    /**
     * The open attempt a return belongs to, taken so it cannot be used again.
     * With `attemptId`, the return must also belong to that attempt.
     *
     * @param {string | null} state
     * @param {string} [attemptId]
     */
    claim(state, attemptId) {
      prune();
      if (!state) return null;
      for (const attempt of attempts.values()) {
        if (attempt.state !== state) continue;
        if (attempt.status !== "pending") return null;
        if (attemptId && attempt.id !== attemptId) return null;
        attempt.status = "working";
        return attempt;
      }
      return null;
    },

    /**
     * Whether an attempt still waits for its return.
     *
     * @param {string} id
     */
    isOpen(id) {
      prune();
      return attempts.get(id)?.status === "pending";
    },

    /** @param {string} id */
    succeed(id) {
      const attempt = attempts.get(id);
      if (attempt) settle(attempt, "done");
    },

    /**
     * @param {string} id
     * @param {string} message
     */
    fail(id, message) {
      const attempt = attempts.get(id);
      if (attempt) settle(attempt, "failed", message);
    },

    /**
     * How an attempt stands, waiting up to `timeoutMs` for it to finish.
     *
     * @param {string} id
     * @param {number} [timeoutMs]
     * @returns {Promise<{ status: "pending" | "done" | "failed", message?: string | null }>}
     */
    async wait(id, timeoutMs = 0) {
      prune();
      const attempt = attempts.get(id);
      if (!attempt || view(attempt).status !== "pending" || timeoutMs <= 0) {
        return view(attempt);
      }
      await new Promise((resolve) => {
        const timer = setTimeout(() => {
          attempt.waiters.delete(wake);
          resolve();
        }, timeoutMs);
        timer.unref?.();
        function wake() {
          clearTimeout(timer);
          resolve();
        }
        attempt.waiters.add(wake);
      });
      // This attempt as it settled: expiring and cancelling also remove it.
      return view(attempt);
    },

    /**
     * A return already being finished checks `cancelled` before it commits
     * anything. A finished sign-in stays finished.
     *
     * @param {string} id
     */
    cancel(id) {
      const attempt = attempts.get(id);
      if (!attempt || attempt.status === "done") return;
      attempt.cancelled = true;
      if (attempt.status === "pending" || attempt.status === "working") {
        settle(attempt, "failed", "Sign-in was cancelled.");
      }
      attempts.delete(id);
    },
  };
}

const RETURN_PAGE_STYLE = `
  body { font-family: ui-sans-serif, system-ui, sans-serif; background: #0b0e14;
         color: #e6e6e6; display: grid; place-items: center; min-height: 100vh; margin: 0; }
  .card { text-align: center; max-width: 26rem; padding: 2rem; }
  h1 { font-size: 1.1rem; font-weight: 600; }
  p { color: #9aa0aa; font-size: .9rem; line-height: 1.5; }`;

const escapeHtml = (text) =>
  String(text).replace(
    /[&<>"']/g,
    (character) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[character],
  );

/**
 * The small page a browser lands on after DigitalOcean. It names the team at
 * most, and never a token.
 *
 * @param {{ ok: boolean, teamName?: string | null, message?: string | null }} result
 */
export function renderReturnPage({ ok, teamName = null, message = null }) {
  const title = ok
    ? teamName
      ? `Signed in to DigitalOcean (${teamName})`
      : "Signed in to DigitalOcean"
    : "Sign-in could not be completed";
  const detail = ok
    ? "You can close this tab and go back to OpenHands."
    : `${message ?? "Something went wrong."} Go back to OpenHands and try again.`;
  return `<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>${escapeHtml(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>${RETURN_PAGE_STYLE}</style>
</head>
<body><div class="card"><h1>${escapeHtml(title)}</h1><p>${escapeHtml(detail)}</p></div>
<script>
  ${
    // A spent code leaves the address bar and history, and a tab Canvas
    // opened may close itself. After a failure the address stays: its code
    // may be unspent (another Canvas on this port), and it can be pasted.
    ok
      ? 'history.replaceState(null, "", location.pathname);\n  setTimeout(function () { window.close(); }, 1500);'
      : ""
  }
</script></body>
</html>`;
}

/**
 * The desktop app's sign-in: open the system browser, take the return on a
 * one-off loopback listener, and trade the code for tokens.
 *
 * The system browser is used rather than an in-app `BrowserWindow` so the user
 * authorizes in the session they are already signed into and can see the real
 * URL they are authenticating against.
 *
 * @param {object} options
 * @param {string | null | undefined} options.clientId
 * @param {(url: string) => Promise<void>} options.openExternal
 * @param {number} [options.timeoutMs]
 * @param {typeof fetch} [options.fetchImpl]
 */
export async function signInWithDigitalOcean({
  clientId,
  openExternal,
  timeoutMs = SIGN_IN_TIMEOUT_MS,
  fetchImpl = globalThis.fetch,
}) {
  if (!clientId) {
    throw new Error(
      "No DigitalOcean OAuth client is configured in this build. Use a personal access token instead.",
    );
  }

  const attempts = createSignInAttempts({ ttlMs: timeoutMs });
  const server = createServer();
  let attemptId = null;

  const returned = new Promise((resolve, reject) => {
    server.on("request", (req, res) => {
      const url = new URL(req.url, "http://127.0.0.1");
      if (req.method !== "GET" || url.pathname !== DESKTOP_CALLBACK_PATH) {
        res.writeHead(404).end();
        return;
      }
      const { code, state, error } = readCallback(url.search);
      const attempt = attempts.claim(state, attemptId);
      if (!attempt) {
        // Not this sign-in's return (a stale tab, a guess): ignore it.
        res.writeHead(400, { "Content-Type": "text/html; charset=utf-8" });
        res.end(
          renderReturnPage({
            ok: false,
            message: "This sign-in is no longer open.",
          }),
        );
        return;
      }
      if (error || !code) {
        res.writeHead(400, { "Content-Type": "text/html; charset=utf-8" });
        res.end(renderReturnPage({ ok: false, message: error }));
        reject(new Error(error ?? "DigitalOcean did not return a code."));
        return;
      }
      exchangeCode({
        clientId,
        code,
        verifier: attempt.verifier,
        redirectUri: attempt.redirectUri,
        fetchImpl,
      }).then(
        (grant) => {
          res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
          res.end(
            renderReturnPage({ ok: true, teamName: grant.account?.name }),
          );
          resolve(grant);
        },
        (failure) => {
          res.writeHead(502, { "Content-Type": "text/html; charset=utf-8" });
          res.end(renderReturnPage({ ok: false, message: failure.message }));
          reject(failure);
        },
      );
    });
  });

  // The race below observes it, but a return can land while openExternal is
  // still resolving; until then its rejection must not count as unhandled.
  returned.catch(() => {});

  const failed = new Promise((_, reject) => server.on("error", reject));

  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(
      () => reject(new Error("Timed out waiting for DigitalOcean sign-in.")),
      timeoutMs,
    );
    timer.unref?.();
  });

  try {
    // Any free port: DigitalOcean matches loopback addresses on any port.
    await Promise.race([
      new Promise((resolve) => server.listen(0, "127.0.0.1", resolve)),
      failed,
    ]);
    const { port } = server.address();
    const begun = attempts.begin({
      clientId,
      redirectUri: `http://127.0.0.1:${port}${DESKTOP_CALLBACK_PATH}`,
    });
    attemptId = begun.attemptId;
    await openExternal(begun.authorizeUrl);
    return await Promise.race([returned, failed, timeout]);
  } finally {
    clearTimeout(timer);
    server.close();
  }
}

/**
 * Invalidate an access token at DigitalOcean, which takes only access
 * tokens here. Best-effort: resolves false when DigitalOcean did not
 * confirm, and a failed revoke must not stop the local sign-in from being
 * forgotten.
 *
 * @param {string} accessToken
 * @param {typeof fetch} [fetchImpl]
 * @returns {Promise<boolean>}
 */
export async function revokeToken(accessToken, fetchImpl = globalThis.fetch) {
  try {
    const response = await fetchImpl(REVOKE_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({ token: accessToken }).toString(),
      signal: AbortSignal.timeout(TOKEN_TIMEOUT_MS),
    });
    return response.ok;
  } catch {
    // Network failure on sign-out; the local record is still removed.
    return false;
  }
}
