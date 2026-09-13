// @vitest-environment node
// Tests for the portal-auth gate wired into scripts/static-server.mjs.
//
// These spin up a real static-server instance with `--portal-auth` enabled and
// exercise the full lifecycle with HTTP requests: unauthenticated redirects to
// /setup, first-admin creation, login, admin user management, session gating of
// static + API routes, logout, and credential/session hashing at rest.
import { request, type Server } from "node:http";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import { startStaticServer } from "../../scripts/static-server.mjs";

describe("static-server portal auth", () => {
  const servers: Server[] = [];
  const tempDirs: string[] = [];
  let nextPort = 41000;

  async function startPortalServer() {
    const dir = mkdtempSync(path.join(tmpdir(), "portal-auth-"));
    tempDirs.push(dir);
    writeFileSync(
      path.join(dir, "index.html"),
      "<html><body>SECRET</body></html>",
    );
    const storePath = path.join(dir, "portal-auth-store.json");
    const port = nextPort++;

    const server = await startStaticServer({
      port,
      host: "127.0.0.1",
      dir,
      routes: {},
      portalAuth: storePath,
    });
    servers.push(server);
    return { base: `http://127.0.0.1:${port}`, storePath, dir };
  }

  afterEach(async () => {
    await Promise.all(
      servers.splice(0).map(
        (server) =>
          new Promise<void>((resolve) => {
            server.close(() => resolve());
          }),
      ),
    );
    for (const dir of tempDirs.splice(0)) {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  function rawRequest(
    url: string,
    method = "GET",
    { body, cookie }: { body?: unknown; cookie?: string } = {},
  ) {
    return new Promise<{
      status: number;
      headers: Record<string, string | string[] | undefined>;
      text: string;
    }>((resolve, reject) => {
      const cookieHeader = cookie
        ? `openhands_portal_session=${cookie}`
        : undefined;
      const req = request(
        url,
        {
          method,
          agent: false,
          headers: {
            ...(cookieHeader ? { Cookie: cookieHeader } : {}),
            ...(body ? { "Content-Type": "application/json" } : {}),
          },
        },
        (res) => {
          let data = "";
          res.on("data", (chunk) => (data += chunk));
          res.on("end", () =>
            resolve({
              status: res.statusCode ?? 0,
              headers: res.headers,
              text: data,
            }),
          );
        },
      );
      req.on("error", reject);
      if (body) req.write(JSON.stringify(body));
      req.end();
    });
  }

  function extractCookie(
    headers: Record<string, string | string[] | undefined>,
  ) {
    const setCookie = headers["set-cookie"];
    if (!setCookie) return null;
    const raw = Array.isArray(setCookie) ? setCookie[0] : setCookie;
    const match = raw.match(/openhands_portal_session=([^;]+)/);
    return match ? match[1] : null;
  }

  it("redirects unauthenticated static requests to /setup before any admin exists", async () => {
    const { base } = await startPortalServer();
    const res = await rawRequest(`${base}/`);
    expect(res.status).toBe(302);
    expect(res.headers.location).toBe("/setup?returnTo=%2F");
  });

  it("serves the setup page and creates the first admin, then gates behind login", async () => {
    const { base } = await startPortalServer();

    const setupPage = await rawRequest(`${base}/setup`);
    expect(setupPage.status).toBe(200);
    expect(setupPage.text).toContain("Set up admin");

    const admin = await rawRequest(`${base}/api/portal-auth/setup`, "POST", {
      body: { username: "grok", password: "supersecret1" },
    });
    expect(admin.status).toBe(201);

    // Admin now exists -> /setup redirects to /login, and re-running setup 409s.
    const after = await rawRequest(`${base}/setup`);
    expect(after.status).toBe(302);
    expect(after.headers.location).toMatch(/^\/login/);

    const dup = await rawRequest(`${base}/api/portal-auth/setup`, "POST", {
      body: { username: "other", password: "supersecret1" },
    });
    expect(dup.status).toBe(409);

    // Logged out, GET / now redirects to /login.
    const root = await rawRequest(`${base}/`);
    expect(root.status).toBe(302);
    expect(root.headers.location).toMatch(/^\/login/);
  });

  it("requires a session cookie for static and API routes, but not for login", async () => {
    const { base } = await startPortalServer();

    await rawRequest(`${base}/api/portal-auth/setup`, "POST", {
      body: { username: "grok", password: "supersecret1" },
    });

    // API path without a session -> 401 JSON, not an HTML redirect.
    const api = await rawRequest(`${base}/api/some-endpoint`);
    expect(api.status).toBe(401);
    expect(JSON.parse(api.text).error).toBe("Authentication required.");

    // Asset without a session -> 401.
    const asset = await rawRequest(`${base}/assets/app.js`);
    expect(asset.status).toBe(401);

    // Login page is reachable without auth.
    const login = await rawRequest(`${base}/login`);
    expect(login.status).toBe(200);
    expect(login.text).toContain("Sign in");

    // Login sets a session cookie and unlocks static content.
    const loginRes = await rawRequest(`${base}/api/portal-auth/login`, "POST", {
      body: { username: "grok", password: "supersecret1" },
    });
    expect(loginRes.status).toBe(200);
    const cookie = extractCookie(loginRes.headers);
    expect(cookie).toBeTruthy();

    const staticRes = await rawRequest(`${base}/`, "GET", { cookie });
    expect(staticRes.status).toBe(200);
    expect(staticRes.text).toContain("SECRET");
  });

  it("rejects bad credentials and enforces admin-only user creation", async () => {
    const { base } = await startPortalServer();

    await rawRequest(`${base}/api/portal-auth/setup`, "POST", {
      body: { username: "grok", password: "supersecret1" },
    });

    const bad = await rawRequest(`${base}/api/portal-auth/login`, "POST", {
      body: { username: "grok", password: "wrongpass1" },
    });
    expect(bad.status).toBe(401);

    const admin = await rawRequest(`${base}/api/portal-auth/login`, "POST", {
      body: { username: "grok", password: "supersecret1" },
    });
    const adminCookie = extractCookie(admin.headers);

    // Admin creates a non-admin user.
    const created = await rawRequest(`${base}/api/portal-auth/users`, "POST", {
      body: { username: "bob", password: "bobspassword" },
      cookie: adminCookie,
    });
    expect(created.status).toBe(201);

    // Non-admin cannot create users.
    const bob = await rawRequest(`${base}/api/portal-auth/login`, "POST", {
      body: { username: "bob", password: "bobspassword" },
    });
    const bobCookie = extractCookie(bob.headers);
    const forbidden = await rawRequest(
      `${base}/api/portal-auth/users`,
      "POST",
      {
        body: { username: "mallory", password: "whatever1" },
        cookie: bobCookie,
      },
    );
    expect(forbidden.status).toBe(403);

    // Unauthenticated user creation -> 401.
    const anon = await rawRequest(`${base}/api/portal-auth/users`, "POST", {
      body: { username: "mallory", password: "whatever1" },
    });
    expect(anon.status).toBe(401);
  });

  it("persists only hashed passwords and hashed sessions in the store", async () => {
    const { base, storePath } = await startPortalServer();

    const setup = await rawRequest(`${base}/api/portal-auth/setup`, "POST", {
      body: { username: "grok", password: "supersecret1" },
    });
    const cookie = extractCookie(setup.headers);
    await rawRequest(`${base}/`, "GET", { cookie });

    const raw = readFileSync(storePath, "utf8");
    expect(raw).not.toContain("supersecret1");
    // Session token must not be stored in the clear.
    expect(raw).not.toContain(cookie);

    const store = JSON.parse(raw);
    expect(store.users.grok.passwordHash).toMatch(/^[0-9a-f]{128}$/);
    expect(store.users.grok.salt).toMatch(/^[0-9a-f]{32}$/);
    // Sessions are keyed by token digest, values carry no raw token.
    expect(
      Object.keys(store.sessions).every((k) => /^[0-9a-f]{64}$/.test(k)),
    ).toBe(true);
  });

  it("logout invalidates the session", async () => {
    const { base } = await startPortalServer();

    await rawRequest(`${base}/api/portal-auth/setup`, "POST", {
      body: { username: "grok", password: "supersecret1" },
    });
    const login = await rawRequest(`${base}/api/portal-auth/login`, "POST", {
      body: { username: "grok", password: "supersecret1" },
    });
    const cookie = extractCookie(login.headers);

    const before = await rawRequest(`${base}/`, "GET", { cookie });
    expect(before.status).toBe(200);

    const logout = await rawRequest(`${base}/api/portal-auth/logout`, "POST", {
      cookie,
    });
    expect(logout.status).toBe(200);

    const after = await rawRequest(`${base}/`, "GET", { cookie });
    expect(after.status).toBe(302);
  });

  it("enforces a minimum password length", async () => {
    const { base } = await startPortalServer();
    const res = await rawRequest(`${base}/api/portal-auth/setup`, "POST", {
      body: { username: "grok", password: "short" },
    });
    expect(res.status).toBe(400);
  });
});
