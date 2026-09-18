/**
 * Self-contained portal auth for the Agent Canvas static server.
 *
 * When `--portal-auth <store>` is passed to scripts/static-server.mjs (or the
 * equivalent flag on scripts/ingress.mjs), every request -- static assets,
 * SPA navigation and proxied API traffic alike -- is gated behind a session
 * cookie. This lets the web UI be exposed publicly without leaving the agent
 * server, automations and conversations open to anyone who can reach the port.
 *
 * Behavior:
 *   - First run (no admin user in the store): every request redirects to
 *     `/setup`. `POST /api/portal-auth/setup` creates the first admin account.
 *     After creation the operator can log in.
 *   - `GET /login` serves a self-contained login page.
 *   - `POST /api/portal-auth/login` validates credentials and sets an
 *     HttpOnly, SameSite=Strict session cookie.
 *   - `POST /api/portal-auth/users` (admin-only) creates additional users.
 *   - `POST /api/portal-auth/logout` clears the session.
 *   - Everything else requires a valid session, otherwise it is redirected to
 *     `/login` (browser navigations) or rejected with 401 (API/asset calls).
 *
 * Passwords are stored as scrypt hashes with a per-user random salt. Sessions
 * are opaque random tokens stored hashed (sha256) in the store with a TTL, so
 * a store leak does not yield usable sessions or passwords.
 *
 * The module has zero dependencies beyond node:crypto and node:fs, so it works
 * in any launcher that already depends on this repo.
 */

import { createServer } from "node:http";
import {
  randomBytes,
  scryptSync,
  timingSafeEqual,
  createHash,
} from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync, renameSync } from "node:fs";
import { dirname, resolve } from "node:path";

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

export const PORTAL_AUTH_SETUP_PATH = "/api/portal-auth/setup";
export const PORTAL_AUTH_LOGIN_PATH = "/api/portal-auth/login";
export const PORTAL_AUTH_LOGOUT_PATH = "/api/portal-auth/logout";
export const PORTAL_AUTH_USERS_PATH = "/api/portal-auth/users";
export const PORTAL_AUTH_SESSION_COOKIE = "openhands_portal_session";
export const PORTAL_AUTH_SESSION_TTL_SECONDS = 12 * 60 * 60; // 12h
export const PORTAL_AUTH_SETUP_CHECK_PATH = "/api/portal-auth/status";

const SCRYPT_KEYLEN = 64;
const SESSION_BYTES = 32;
const SALT_BYTES = 16;

function hashPassword(password, saltHex) {
  const salt = Buffer.from(saltHex, "hex");
  return scryptSync(password, salt, SCRYPT_KEYLEN).toString("hex");
}

function randomToken() {
  return randomBytes(SESSION_BYTES).toString("hex");
}

function tokenDigest(token) {
  return createHash("sha256").update(token).digest("hex");
}

// ─────────────────────────────────────────────────────────────────────────────
// Store
// ─────────────────────────────────────────────────────────────────────────────

function defaultStoreShape() {
  return {
    version: 1,
    users: {}, // username -> { salt, passwordHash, isAdmin, createdAt }
    sessions: {}, // tokenDigest -> { username, expiresAt }
  };
}

/**
 * A JSON-file backed credential + session store with atomic-ish writes
 * (write temp file, then rename over the target).
 */
export class PortalAuthStore {
  constructor(filePath) {
    this.filePath = resolve(filePath);
    this.data = this.#load();
    // Scrub any sessions that already expired at rest to avoid unbounded growth.
    this.#pruneExpired();
  }

  #load() {
    try {
      const raw = readFileSync(this.filePath, "utf8");
      const parsed = JSON.parse(raw);
      const base = defaultStoreShape();
      return {
        ...base,
        ...parsed,
        users: { ...base.users, ...(parsed.users ?? {}) },
        sessions: { ...(parsed.sessions ?? {}) },
      };
    } catch {
      return defaultStoreShape();
    }
  }

  #persist() {
    mkdirSync(dirname(this.filePath), { recursive: true });
    const tmp = `${this.filePath}.${process.pid}.tmp`;
    writeFileSync(tmp, JSON.stringify(this.data, null, 2), { mode: 0o600 });
    renameSync(tmp, this.filePath);
  }

  #pruneExpired() {
    const now = Date.now();
    let changed = false;
    for (const [digest, session] of Object.entries(this.data.sessions)) {
      if (session.expiresAt <= now) {
        delete this.data.sessions[digest];
        changed = true;
      }
    }
    if (changed) this.#persist();
  }

  get hasAdmin() {
    return Object.values(this.data.users).some((u) => u.isAdmin);
  }

  listUsers() {
    return Object.entries(this.data.users).map(([username, u]) => ({
      username,
      isAdmin: Boolean(u.isAdmin),
      createdAt: u.createdAt,
    }));
  }

  createUser(username, password, { isAdmin = false } = {}) {
    const name = username.trim();
    if (!name || !password)
      throw new PortalAuthError(400, "Username and password are required.");
    if (name.length > 64) throw new PortalAuthError(400, "Username too long.");
    if (password.length < 8) {
      throw new PortalAuthError(400, "Password must be at least 8 characters.");
    }
    if (this.data.users[name]) {
      throw new PortalAuthError(409, "User already exists.");
    }
    const salt = randomBytes(SALT_BYTES).toString("hex");
    this.data.users[name] = {
      salt,
      passwordHash: hashPassword(password, salt),
      isAdmin: Boolean(isAdmin),
      createdAt: new Date().toISOString(),
    };
    this.#persist();
    return { username: name, isAdmin: Boolean(isAdmin) };
  }

  verifyUser(username) {
    const user = this.data.users[username];
    if (!user) return null;
    return { username, isAdmin: user.isAdmin };
  }

  authenticate(username, password) {
    const user = this.data.users[username];
    if (!user) return false;
    const candidate = hashPassword(password, user.salt);
    const stored = Buffer.from(user.passwordHash, "hex");
    const provided = Buffer.from(candidate, "hex");
    if (stored.length !== provided.length) return false;
    return timingSafeEqual(stored, provided);
  }

  createSession(username) {
    const token = randomToken();
    const digest = tokenDigest(token);
    this.data.sessions[digest] = {
      username,
      expiresAt: Date.now() + PORTAL_AUTH_SESSION_TTL_SECONDS * 1000,
    };
    this.#persist();
    return token;
  }

  /**
   * @returns {string | null} username for a valid session token, else null.
   */
  resolveSession(token) {
    if (!token) return null;
    const session = this.data.sessions[tokenDigest(token)];
    if (!session) return null;
    if (session.expiresAt <= Date.now()) {
      delete this.data.sessions[tokenDigest(token)];
      this.#persist();
      return null;
    }
    return session.username;
  }

  destroySession(token) {
    if (!token) return;
    if (this.data.sessions[tokenDigest(token)]) {
      delete this.data.sessions[tokenDigest(token)];
      this.#persist();
    }
  }
}

export class PortalAuthError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP helpers
// ─────────────────────────────────────────────────────────────────────────────

function readBody(req, limit = 64 * 1024) {
  return new Promise((resolvePromise, reject) => {
    let size = 0;
    const chunks = [];
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > limit) {
        reject(new PortalAuthError(413, "Request body too large."));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      try {
        const raw = Buffer.concat(chunks).toString("utf8");
        resolvePromise(raw ? JSON.parse(raw) : {});
      } catch {
        reject(new PortalAuthError(400, "Invalid JSON body."));
      }
    });
    req.on("error", reject);
  });
}

function sendJson(res, status, payload, extraHeaders = {}) {
  const body = Buffer.from(JSON.stringify(payload), "utf8");
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": body.length,
    "Cache-Control": "no-store",
    ...extraHeaders,
  });
  res.end(body);
}

function sendHtml(res, html) {
  const body = Buffer.from(html, "utf8");
  res.writeHead(200, {
    "Content-Type": "text/html; charset=utf-8",
    "Content-Length": body.length,
    "Cache-Control": "no-store",
  });
  res.end(body);
}

function setSessionCookie(res, token) {
  res.setHeader(
    "Set-Cookie",
    `${PORTAL_AUTH_SESSION_COOKIE}=${token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=${PORTAL_AUTH_SESSION_TTL_SECONDS}`,
  );
}

function clearSessionCookie(res) {
  res.setHeader(
    "Set-Cookie",
    `${PORTAL_AUTH_SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0`,
  );
}

function parseCookies(req) {
  const header = req.headers.cookie;
  if (!header) return {};
  const out = {};
  for (const part of header.split(";")) {
    const idx = part.indexOf("=");
    if (idx < 0) continue;
    const key = part.slice(0, idx).trim();
    const value = part.slice(idx + 1).trim();
    out[key] = value;
  }
  return out;
}

function isHtmlNavigation(req, urlPath) {
  if (req.method !== "GET" && req.method !== "HEAD") return false;
  // Asset paths and explicitly fetched files (API responses) get a 401, not a
  // redirect, so XHR and asset loads fail cleanly instead of returning the
  // login page markup.
  const last = urlPath.split("/").pop() ?? "";
  const dot = last.lastIndexOf(".");
  const ext = dot >= 0 ? last.slice(dot + 1).toLowerCase() : "";
  if (ext && ["html", "htm"].includes(ext)) return true;
  if (ext) return false; // has an extension -> asset
  return !urlPath.startsWith("/api/");
}

// ─────────────────────────────────────────────────────────────────────────────
// Pages
// ─────────────────────────────────────────────────────────────────────────────

const PAGE_STYLE = `
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #0b0f19; color: #e6e9f0; min-height: 100vh; display: grid; place-items: center; }
  .card { background: #141a2a; border: 1px solid #2a3350; border-radius: 12px;
          padding: 32px 28px; width: min(360px, 92vw); }
  h1 { margin: 0 0 4px; font-size: 20px; }
  p.sub { margin: 0 0 20px; color: #9aa3b8; font-size: 13px; }
  label { display: block; font-size: 13px; margin: 12px 0 6px; color: #b9c1d4; }
  input { width: 100%; padding: 10px 12px; border-radius: 8px; border: 1px solid #2f3a5c;
          background: #0e1424; color: #e6e9f0; font-size: 14px; }
  input:focus { outline: none; border-color: #5b7cfa; }
  button { width: 100%; margin-top: 18px; padding: 11px; border: 0; border-radius: 8px;
           background: #5b7cfa; color: #fff; font-size: 14px; font-weight: 600; cursor: pointer; }
  button:hover { background: #4a6bf0; }
  .error { margin-top: 14px; color: #ff7a85; font-size: 13px; min-height: 18px; white-space: pre-wrap; }
  .ok { margin-top: 14px; color: #58d68d; font-size: 13px; }
  small.hint { color: #8a93ab; font-size: 12px; line-height: 1.5; }
  .nav { margin-top: 18px; text-align: center; }
  .nav a { color: #9ab0ff; font-size: 13px; text-decoration: none; }
`;

function shellPage({ title, subtitle, body, script }) {
  return `<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>${title}</title>
<style>${PAGE_STYLE}</style>
</head><body>
<div class="card">
  <h1>${title}</h1>
  <p class="sub">${subtitle}</p>
  ${body}
</div>
<script>${script}</script>
</body></html>`;
}

function loginPageHtml() {
  const form = `
  <form id="f">
    <label for="username">Username</label>
    <input id="username" name="username" autocomplete="username" required autofocus/>
    <label for="password">Password</label>
    <input id="password" name="password" type="password" autocomplete="current-password" required/>
    <button type="submit">Sign in</button>
  </form>
  <div class="error" id="err"></div>`;
  const script = `
    const f = document.getElementById('f');
    f.addEventListener('submit', async (e) => {
      e.preventDefault();
      const err = document.getElementById('err'); err.textContent = '';
      try {
        const r = await fetch('/api/portal-auth/login', {
          method: 'POST', headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ username: f.username.value, password: f.password.value })
        });
        const data = await r.json();
        if (!r.ok) { err.textContent = data.error || 'Login failed.'; return; }
        window.location.href = data.returnTo || '/';
      } catch { err.textContent = 'Network error.'; }
    });
    document.addEventListener('DOMContentLoaded', () => f.username.focus());
  `;
  return shellPage({
    title: "Sign in",
    subtitle: "Agent Canvas portal",
    body: form,
    script,
  });
}

function setupPageHtml() {
  const form = `
  <div class="ok" id="hint">No admin account configured yet. Create one to lock down this instance.</div>
  <form id="f">
    <label for="username">Admin username</label>
    <input id="username" name="username" autocomplete="username" required autofocus/>
    <label for="password">Password</label>
    <input id="password" name="password" type="password" autocomplete="new-password" required minlength="8"/>
    <small class="hint">At least 8 characters. This account can create additional users later.</small>
    <button type="submit">Create admin</button>
  </form>
  <div class="error" id="err"></div>`;
  const script = `
    const f = document.getElementById('f');
    f.addEventListener('submit', async (e) => {
      e.preventDefault();
      const err = document.getElementById('err'); err.textContent = '';
      try {
        const r = await fetch('/api/portal-auth/setup', {
          method: 'POST', headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ username: f.username.value, password: f.password.value })
        });
        const data = await r.json();
        if (!r.ok) { err.textContent = data.error || 'Failed to create admin.'; return; }
        window.location.href = data.returnTo || '/';
      } catch { err.textContent = 'Network error.'; }
    });
    document.addEventListener('DOMContentLoaded', () => f.username.focus());
  `;
  return shellPage({
    title: "Set up admin",
    subtitle: "First-time configuration",
    body: form,
    script,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Request handler that front-runs the static/proxy router
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Handle a portal-auth request. Returns `true` if the request was fully
 * handled (response written), otherwise `false` to signal the caller should
 * continue serving the request normally (i.e. it is authenticated).
 *
 * @param {import("node:http").ClientRequest} req
 * @param {import("node:http").ServerResponse} res
 * @param {PortalAuthStore} store
 * @param {{ loginPath?: string, returnTo?: string }} [opts]
 */
export function createPortalAuthHandler(store, opts = {}) {
  const loginPath = opts.loginPath ?? "/login";

  return function handlePortalAuth(req, res) {
    const urlPath = (req.url ?? "/").split("?")[0];
    const cookies = parseCookies(req);
    const sessionToken = cookies[PORTAL_AUTH_SESSION_COOKIE];
    const username = store.resolveSession(sessionToken);

    // ── Auth API endpoints ──────────────────────────────────────────────────
    if (req.method === "POST" && urlPath === PORTAL_AUTH_SETUP_PATH) {
      if (store.hasAdmin) {
        sendJson(res, 409, { error: "Admin already configured." });
        return true;
      }
      return readBody(req)
        .then((body) => {
          store.createUser(
            String(body.username ?? ""),
            String(body.password ?? ""),
            {
              isAdmin: true,
            },
          );
          const token = store.createSession(String(body.username).trim());
          setSessionCookie(res, token);
          sendJson(res, 201, { ok: true, returnTo: opts.returnTo ?? "/" });
        })
        .catch((err) => {
          const status = err instanceof PortalAuthError ? err.status : 500;
          sendJson(res, status, { error: err.message ?? "Internal error." });
        })
        .then(() => true);
    }

    if (req.method === "POST" && urlPath === PORTAL_AUTH_LOGIN_PATH) {
      return readBody(req)
        .then((body) => {
          const name = String(body.username ?? "").trim();
          const password = String(body.password ?? "");
          if (!store.authenticate(name, password)) {
            throw new PortalAuthError(401, "Invalid username or password.");
          }
          const token = store.createSession(name);
          setSessionCookie(res, token);
          const returnTo = String(body.returnTo ?? opts.returnTo ?? "/");
          sendJson(res, 200, { ok: true, returnTo });
        })
        .catch((err) => {
          const status = err instanceof PortalAuthError ? err.status : 500;
          sendJson(res, status, { error: err.message ?? "Internal error." });
        })
        .then(() => true);
    }

    if (req.method === "POST" && urlPath === PORTAL_AUTH_LOGOUT_PATH) {
      store.destroySession(sessionToken);
      clearSessionCookie(res);
      sendJson(res, 200, { ok: true });
      return true;
    }

    if (req.method === "GET" && urlPath === PORTAL_AUTH_SETUP_CHECK_PATH) {
      sendJson(res, 200, { configured: store.hasAdmin });
      return true;
    }

    if (req.method === "POST" && urlPath === PORTAL_AUTH_USERS_PATH) {
      const admin = username ? store.verifyUser(username) : null;
      if (!admin) {
        sendJson(res, 401, { error: "Authentication required." });
        return true;
      }
      if (!admin.isAdmin) {
        sendJson(res, 403, { error: "Admin role required." });
        return true;
      }
      return readBody(req)
        .then((body) => {
          const created = store.createUser(
            String(body.username ?? ""),
            String(body.password ?? ""),
            { isAdmin: Boolean(body.isAdmin) },
          );
          sendJson(res, 201, { ok: true, user: created });
        })
        .catch((err) => {
          const status = err instanceof PortalAuthError ? err.status : 500;
          sendJson(res, status, {
            error: err.message ?? "Failed to create user.",
          });
        })
        .then(() => true);
    }

    // ── Login / setup pages ───────────────────────────────────────────────────
    if (
      (req.method === "GET" || req.method === "HEAD") &&
      urlPath === loginPath
    ) {
      sendHtml(res, loginPageHtml());
      return true;
    }
    if (urlPath === "/setup" && !store.hasAdmin) {
      sendHtml(res, setupPageHtml());
      return true;
    }

    // ── Login / logout actions ────────────────────────────────────────────────
    if (req.method === "POST" && urlPath === PORTAL_AUTH_LOGIN_PATH) {
      return readBody(req)
        .then((body) => {
          const uname = String(body.username ?? "").trim();
          const pass = String(body.password ?? "");
          if (!store.authenticate(uname, pass)) {
            sendJson(res, 401, { error: "Invalid username or password." });
            return true;
          }
          const token = store.createSession(uname);
          setSessionCookie(res, token);
          sendJson(res, 200, { ok: true, returnTo: opts.returnTo ?? "/" });
          return true;
        })
        .catch((err) => {
          const status = err instanceof PortalAuthError ? err.status : 500;
          sendJson(res, status, { error: err.message ?? "Login failed." });
          return true;
        });
    }

    if (req.method === "POST" && urlPath === PORTAL_AUTH_LOGOUT_PATH) {
      store.destroySession(sessionToken);
      clearSessionCookie(res);
      sendJson(res, 200, { ok: true });
      return true;
    }

    // ── Gating ───────────────────────────────────────────────────────────────
    if (!username) {
      const isNav = isHtmlNavigation(req, urlPath);
      if (isNav || urlPath.startsWith("/setup")) {
        const target = store.hasAdmin ? loginPath : "/setup";
        res.writeHead(302, {
          Location: `${target}?returnTo=${encodeURIComponent(urlPath)}`,
        });
        res.end();
      } else {
        sendJson(res, 401, { error: "Authentication required." });
      }
      return true; // blocked
    }

    return false; // authenticated -> continue serving
  };
}

// Exported mostly for tests / parity with ingress.
export const _internal = {
  hashPassword,
  randomToken,
  tokenDigest,
  parseCookies,
  isHtmlNavigation,
  defaultStoreShape,
};

// ─────────────────────────────────────────────────────────────────────────────
// Standalone dev server (start with: node scripts/portal-auth.mjs --store <path>)
// ─────────────────────────────────────────────────────────────────────────────

if (process.argv[1] && import.meta.url === new URL(import.meta.url).pathname) {
  const storePathIndex = process.argv.indexOf("--store");
  const storePath =
    storePathIndex >= 0
      ? process.argv[storePathIndex + 1]
      : ".portal-auth.json";
  const store = new PortalAuthStore(storePath);
  const handler = createPortalAuthHandler(store);
  const server = createServer((req, res) => {
    if (!handler(req, res)) {
      res.writeHead(200, { "Content-Type": "text/plain" });
      res.end("authenticated");
    }
  });
  server.listen(3005, "::", () => {
    console.log(`portal-auth dev server on :3005 (store: ${store.filePath})`);
  });
}
