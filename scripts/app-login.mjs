/**
 * Internal app login (username/password) for Agent Canvas.
 *
 * Persist users under the canvas state dir with bcrypt password hashes.
 * Sessions are httpOnly cookies remembered across browser restarts.
 *
 * Env:
 *   APP_LOGIN_ENABLED — opt-in; set true/1/yes/on to enable the UI gate
 *   OH_CANVAS_SAFE_STATE_DIR — override state directory
 */

import { createHash, randomBytes } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import bcrypt from "bcryptjs";

export const APP_LOGIN_PATH_PREFIX = "/api/app-login";
export const APP_LOGIN_COOKIE_NAME = "oh_app_login_session";
export const DEFAULT_APP_LOGIN_USERNAME = "heimdallsec";
export const DEFAULT_APP_LOGIN_PASSWORD = "heimdallsec";
export const APP_LOGIN_SESSION_MAX_AGE_SEC = 60 * 60 * 24 * 30; // 30 days
const BCRYPT_ROUNDS = 10;
const USERS_FILENAME = "app-users.json";
const SESSIONS_FILENAME = "app-login-sessions.json";

/**
 * @param {string | undefined} value
 * @returns {boolean}
 */
export function isAppLoginEnabled(value = process.env.APP_LOGIN_ENABLED) {
  if (value === undefined || value === null || String(value).trim() === "") {
    return false;
  }
  const normalized = String(value).trim().toLowerCase();
  return ["1", "true", "yes", "on"].includes(normalized);
}

/**
 * @param {NodeJS.ProcessEnv} [env]
 */
export function resolveAppLoginStateDir(env = process.env) {
  return (
    env.OH_CANVAS_SAFE_STATE_DIR ||
    join(homedir(), ".openhands", "agent-canvas")
  );
}

/**
 * @param {string} url
 */
export function isAppLoginRequest(url) {
  const path = (url ?? "/").split("?")[0];
  return (
    path === APP_LOGIN_PATH_PREFIX ||
    path.startsWith(`${APP_LOGIN_PATH_PREFIX}/`)
  );
}

/**
 * @param {string} password
 */
export async function hashAppLoginPassword(password) {
  return bcrypt.hash(password, BCRYPT_ROUNDS);
}

/**
 * @param {string} password
 * @param {string} hash
 */
export async function verifyAppLoginPassword(password, hash) {
  return bcrypt.compare(password, hash);
}

/**
 * @param {string} token
 */
function hashSessionToken(token) {
  return createHash("sha256").update(token).digest("hex");
}

/**
 * @param {{ stateDir?: string, enabled?: boolean }} [options]
 */
export function createAppLoginStore(options = {}) {
  const stateDir = options.stateDir ?? resolveAppLoginStateDir();
  const enabled =
    options.enabled === undefined
      ? isAppLoginEnabled()
      : Boolean(options.enabled);

  const usersPath = join(stateDir, USERS_FILENAME);
  const sessionsPath = join(stateDir, SESSIONS_FILENAME);

  /** @type {{ users: Array<{ username: string, passwordHash: string }> } | null} */
  let usersCache = null;
  /** @type {{ sessions: Record<string, { username: string, createdAt: string }> } | null} */
  let sessionsCache = null;

  async function ensureStateDir() {
    await mkdir(stateDir, { recursive: true });
  }

  async function loadUsers() {
    if (usersCache) return usersCache;
    await ensureStateDir();
    try {
      const raw = await readFile(usersPath, "utf8");
      const parsed = JSON.parse(raw);
      if (!parsed || !Array.isArray(parsed.users)) {
        throw new Error("invalid users file");
      }
      usersCache = {
        users: parsed.users
          .filter(
            (u) =>
              u &&
              typeof u.username === "string" &&
              typeof u.passwordHash === "string",
          )
          .map((u) => ({
            username: u.username,
            passwordHash: u.passwordHash,
          })),
      };
    } catch {
      // Missing or corrupt file → seed the default internal user.
      const passwordHash = await hashAppLoginPassword(DEFAULT_APP_LOGIN_PASSWORD);
      usersCache = {
        users: [
          {
            username: DEFAULT_APP_LOGIN_USERNAME,
            passwordHash,
          },
        ],
      };
      await persistUsers();
    }
    if (usersCache.users.length === 0) {
      const passwordHash = await hashAppLoginPassword(DEFAULT_APP_LOGIN_PASSWORD);
      usersCache.users.push({
        username: DEFAULT_APP_LOGIN_USERNAME,
        passwordHash,
      });
      await persistUsers();
    }
    return usersCache;
  }

  async function persistUsers() {
    await ensureStateDir();
    await writeFile(usersPath, `${JSON.stringify(usersCache, null, 2)}\n`, {
      mode: 0o600,
    });
  }

  async function loadSessions() {
    if (sessionsCache) return sessionsCache;
    await ensureStateDir();
    try {
      const raw = await readFile(sessionsPath, "utf8");
      const parsed = JSON.parse(raw);
      sessionsCache = {
        sessions:
          parsed && typeof parsed.sessions === "object" && parsed.sessions
            ? parsed.sessions
            : {},
      };
    } catch {
      sessionsCache = { sessions: {} };
      await persistSessions();
    }
    return sessionsCache;
  }

  async function persistSessions() {
    await ensureStateDir();
    await writeFile(
      sessionsPath,
      `${JSON.stringify(sessionsCache, null, 2)}\n`,
      { mode: 0o600 },
    );
  }

  return {
    enabled,
    stateDir,
    usersPath,
    sessionsPath,

    async listUsernames() {
      const data = await loadUsers();
      return data.users.map((u) => u.username).sort((a, b) => a.localeCompare(b));
    },

    async authenticate(username, password) {
      const data = await loadUsers();
      const user = data.users.find((u) => u.username === username);
      if (!user) {
        // Compare against a dummy hash to reduce timing variance on missing users.
        await verifyAppLoginPassword(
          password,
          "$2a$10$abcdefghijklmnopqrstuuABCDEFGHIJKLMNOPQRSTUVWX",
        );
        return false;
      }
      return verifyAppLoginPassword(password, user.passwordHash);
    },

    async createSession(username) {
      const token = randomBytes(32).toString("hex");
      const tokenHash = hashSessionToken(token);
      const data = await loadSessions();
      data.sessions[tokenHash] = {
        username,
        createdAt: new Date().toISOString(),
      };
      await persistSessions();
      return token;
    },

    async resolveSession(token) {
      if (!token) return null;
      const data = await loadSessions();
      const entry = data.sessions[hashSessionToken(token)];
      if (!entry?.username) return null;
      const users = await loadUsers();
      if (!users.users.some((u) => u.username === entry.username)) {
        delete data.sessions[hashSessionToken(token)];
        await persistSessions();
        return null;
      }
      return entry.username;
    },

    async destroySession(token) {
      if (!token) return;
      const data = await loadSessions();
      delete data.sessions[hashSessionToken(token)];
      await persistSessions();
    },

    async addUser(username, password) {
      const normalized = username.trim();
      if (!normalized) {
        throw Object.assign(new Error("Username is required"), { status: 400 });
      }
      if (!password || password.length < 4) {
        throw Object.assign(new Error("Password must be at least 4 characters"), {
          status: 400,
        });
      }
      const data = await loadUsers();
      if (data.users.some((u) => u.username === normalized)) {
        throw Object.assign(new Error("Username already exists"), { status: 409 });
      }
      data.users.push({
        username: normalized,
        passwordHash: await hashAppLoginPassword(password),
      });
      await persistUsers();
      return normalized;
    },

    async removeUser(username) {
      const data = await loadUsers();
      if (data.users.length <= 1) {
        throw Object.assign(new Error("Cannot delete the last user"), {
          status: 400,
        });
      }
      const next = data.users.filter((u) => u.username !== username);
      if (next.length === data.users.length) {
        throw Object.assign(new Error("User not found"), { status: 404 });
      }
      data.users = next;
      await persistUsers();

      const sessions = await loadSessions();
      for (const [hash, entry] of Object.entries(sessions.sessions)) {
        if (entry.username === username) {
          delete sessions.sessions[hash];
        }
      }
      await persistSessions();
    },

    /** @internal test helper */
    async _resetCaches() {
      usersCache = null;
      sessionsCache = null;
    },
  };
}

/**
 * @param {import("node:http").IncomingMessage} req
 */
export function parseCookies(req) {
  const header = req.headers.cookie;
  /** @type {Record<string, string>} */
  const out = {};
  if (!header) return out;
  for (const part of header.split(";")) {
    const idx = part.indexOf("=");
    if (idx < 0) continue;
    const key = part.slice(0, idx).trim();
    const value = part.slice(idx + 1).trim();
    if (!key) continue;
    try {
      out[key] = decodeURIComponent(value);
    } catch {
      out[key] = value;
    }
  }
  return out;
}

/**
 * @param {string} token
 * @param {{ maxAgeSec?: number, clear?: boolean }} [opts]
 */
export function buildAppLoginSetCookie(token, opts = {}) {
  const maxAge = opts.clear ? 0 : (opts.maxAgeSec ?? APP_LOGIN_SESSION_MAX_AGE_SEC);
  const value = opts.clear ? "" : encodeURIComponent(token);
  return [
    `${APP_LOGIN_COOKIE_NAME}=${value}`,
    "Path=/",
    `Max-Age=${maxAge}`,
    "HttpOnly",
    "SameSite=Lax",
  ].join("; ");
}

/**
 * @param {import("node:http").IncomingMessage} req
 */
async function readJsonBody(req) {
  const chunks = [];
  for await (const chunk of req) {
    chunks.push(chunk);
  }
  if (chunks.length === 0) return {};
  const raw = Buffer.concat(chunks).toString("utf8");
  if (!raw.trim()) return {};
  try {
    return JSON.parse(raw);
  } catch {
    throw Object.assign(new Error("Invalid JSON body"), { status: 400 });
  }
}

/**
 * @param {import("node:http").ServerResponse} res
 * @param {number} status
 * @param {unknown} body
 * @param {Record<string, string>} [headers]
 */
function sendJson(res, status, body, headers = {}) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    ...headers,
  });
  res.end(payload);
}

/**
 * @param {import("node:http").IncomingMessage} req
 * @param {ReturnType<typeof createAppLoginStore>} store
 */
async function requireSessionUser(req, store) {
  const cookies = parseCookies(req);
  const token = cookies[APP_LOGIN_COOKIE_NAME];
  const username = await store.resolveSession(token);
  if (!username) {
    throw Object.assign(new Error("Not authenticated"), { status: 401 });
  }
  return { username, token };
}

/**
 * Create a request handler for /api/app-login/*.
 *
 * @param {ReturnType<typeof createAppLoginStore>} [store]
 */
export function createAppLoginHandler(store = createAppLoginStore()) {
  return async function handleAppLogin(req, res) {
    const url = new URL(req.url ?? "/", "http://localhost");
    const path = url.pathname;
    const method = req.method ?? "GET";

    if (!isAppLoginRequest(path)) {
      return false;
    }

    try {
      if (path === `${APP_LOGIN_PATH_PREFIX}/status` && method === "GET") {
        sendJson(res, 200, { enabled: store.enabled });
        return true;
      }

      if (!store.enabled) {
        sendJson(res, 404, { enabled: false, error: "App login is disabled" });
        return true;
      }

      if (path === `${APP_LOGIN_PATH_PREFIX}/me` && method === "GET") {
        const cookies = parseCookies(req);
        const username = await store.resolveSession(
          cookies[APP_LOGIN_COOKIE_NAME],
        );
        if (!username) {
          sendJson(res, 401, { authenticated: false });
          return true;
        }
        sendJson(res, 200, { authenticated: true, username });
        return true;
      }

      if (path === `${APP_LOGIN_PATH_PREFIX}/login` && method === "POST") {
        const body = await readJsonBody(req);
        const username =
          typeof body.username === "string" ? body.username.trim() : "";
        const password = typeof body.password === "string" ? body.password : "";
        if (!username || !password) {
          sendJson(res, 400, { error: "Username and password are required" });
          return true;
        }
        const ok = await store.authenticate(username, password);
        if (!ok) {
          sendJson(res, 401, { error: "Invalid username or password" });
          return true;
        }
        const token = await store.createSession(username);
        sendJson(
          res,
          200,
          { authenticated: true, username },
          { "Set-Cookie": buildAppLoginSetCookie(token) },
        );
        return true;
      }

      if (path === `${APP_LOGIN_PATH_PREFIX}/logout` && method === "POST") {
        const cookies = parseCookies(req);
        await store.destroySession(cookies[APP_LOGIN_COOKIE_NAME]);
        sendJson(
          res,
          200,
          { authenticated: false },
          { "Set-Cookie": buildAppLoginSetCookie("", { clear: true }) },
        );
        return true;
      }

      if (path === `${APP_LOGIN_PATH_PREFIX}/users` && method === "GET") {
        await requireSessionUser(req, store);
        sendJson(res, 200, {
          users: (await store.listUsernames()).map((username) => ({ username })),
        });
        return true;
      }

      if (path === `${APP_LOGIN_PATH_PREFIX}/users` && method === "POST") {
        await requireSessionUser(req, store);
        const body = await readJsonBody(req);
        const username =
          typeof body.username === "string" ? body.username : "";
        const password =
          typeof body.password === "string" ? body.password : "";
        const created = await store.addUser(username, password);
        sendJson(res, 201, { username: created });
        return true;
      }

      const deleteMatch = path.match(
        new RegExp(`^${APP_LOGIN_PATH_PREFIX}/users/([^/]+)$`),
      );
      if (deleteMatch && method === "DELETE") {
        await requireSessionUser(req, store);
        const target = decodeURIComponent(deleteMatch[1]);
        await store.removeUser(target);
        sendJson(res, 200, { deleted: target });
        return true;
      }

      sendJson(res, 404, { error: "Not found" });
      return true;
    } catch (err) {
      const status =
        err && typeof err === "object" && "status" in err
          ? Number(err.status) || 500
          : 500;
      const message =
        err instanceof Error ? err.message : "Internal Server Error";
      sendJson(res, status, { error: message });
      return true;
    }
  };
}
