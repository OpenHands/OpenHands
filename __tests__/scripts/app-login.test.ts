import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  APP_LOGIN_COOKIE_NAME,
  APP_LOGIN_PATH_PREFIX,
  DEFAULT_APP_LOGIN_PASSWORD,
  DEFAULT_APP_LOGIN_USERNAME,
  createAppLoginHandler,
  createAppLoginStore,
  isAppLoginEnabled,
  isAppLoginRequest,
  verifyAppLoginPassword,
} from "../../scripts/app-login.mjs";

describe("isAppLoginEnabled", () => {
  it.each([
    [undefined, false],
    ["", false],
    ["true", true],
    ["1", true],
    ["yes", true],
    ["on", true],
    ["false", false],
    ["0", false],
    ["no", false],
    ["off", false],
  ])("maps %j → %s", (value, expected) => {
    expect(isAppLoginEnabled(value)).toBe(expected);
  });
});

describe("isAppLoginRequest", () => {
  it("matches the app-login API prefix", () => {
    expect(isAppLoginRequest(`${APP_LOGIN_PATH_PREFIX}/status`)).toBe(true);
    expect(isAppLoginRequest(`${APP_LOGIN_PATH_PREFIX}/login?x=1`)).toBe(true);
    expect(isAppLoginRequest("/api/settings")).toBe(false);
  });
});

describe("createAppLoginStore + handler", () => {
  let stateDir: string;
  let store: ReturnType<typeof createAppLoginStore>;
  let handle: ReturnType<typeof createAppLoginHandler>;

  beforeEach(async () => {
    stateDir = await mkdtemp(join(tmpdir(), "app-login-"));
    store = createAppLoginStore({ stateDir, enabled: true });
    handle = createAppLoginHandler(store);
  });

  afterEach(async () => {
    await rm(stateDir, { recursive: true, force: true });
  });

  async function request(
    method: string,
    path: string,
    options: {
      body?: unknown;
      cookie?: string;
    } = {},
  ) {
    const server = createServer((req, res) => {
      void handle(req, res);
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const address = server.address();
    if (!address || typeof address === "string") {
      throw new Error("expected TCP address");
    }

    try {
      const headers: Record<string, string> = {
        Accept: "application/json",
      };
      if (options.cookie) headers.Cookie = options.cookie;
      let body: string | undefined;
      if (options.body !== undefined) {
        body = JSON.stringify(options.body);
        headers["Content-Type"] = "application/json";
      }

      const response = await fetch(
        `http://127.0.0.1:${address.port}${path}`,
        { method, headers, body },
      );
      const setCookie = response.headers.getSetCookie?.() ?? [];
      const json = (await response.json()) as Record<string, unknown>;
      return { status: response.status, json, setCookie };
    } finally {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    }
  }

  function sessionCookieFrom(setCookie: string[]) {
    const line = setCookie.find((c) => c.startsWith(`${APP_LOGIN_COOKIE_NAME}=`));
    expect(line).toBeTruthy();
    return line!.split(";")[0];
  }

  it("seeds the default heimdallsec user with a bcrypt hash", async () => {
    const usernames = await store.listUsernames();
    expect(usernames).toEqual([DEFAULT_APP_LOGIN_USERNAME]);
    expect(
      await store.authenticate(
        DEFAULT_APP_LOGIN_USERNAME,
        DEFAULT_APP_LOGIN_PASSWORD,
      ),
    ).toBe(true);
    expect(await store.authenticate(DEFAULT_APP_LOGIN_USERNAME, "wrong")).toBe(
      false,
    );
  });

  it("reports enabled status without auth", async () => {
    const res = await request("GET", `${APP_LOGIN_PATH_PREFIX}/status`);
    expect(res.status).toBe(200);
    expect(res.json).toEqual({ enabled: true });
  });

  it("logs in with default credentials and remembers the session cookie", async () => {
    const login = await request("POST", `${APP_LOGIN_PATH_PREFIX}/login`, {
      body: {
        username: DEFAULT_APP_LOGIN_USERNAME,
        password: DEFAULT_APP_LOGIN_PASSWORD,
      },
    });
    expect(login.status).toBe(200);
    expect(login.json).toEqual({
      authenticated: true,
      username: DEFAULT_APP_LOGIN_USERNAME,
    });
    const cookie = sessionCookieFrom(login.setCookie);

    const me = await request("GET", `${APP_LOGIN_PATH_PREFIX}/me`, { cookie });
    expect(me.status).toBe(200);
    expect(me.json).toEqual({
      authenticated: true,
      username: DEFAULT_APP_LOGIN_USERNAME,
    });
  });

  it("rejects invalid credentials", async () => {
    const res = await request("POST", `${APP_LOGIN_PATH_PREFIX}/login`, {
      body: { username: DEFAULT_APP_LOGIN_USERNAME, password: "nope" },
    });
    expect(res.status).toBe(401);
  });

  it("adds and lists users after login", async () => {
    const login = await request("POST", `${APP_LOGIN_PATH_PREFIX}/login`, {
      body: {
        username: DEFAULT_APP_LOGIN_USERNAME,
        password: DEFAULT_APP_LOGIN_PASSWORD,
      },
    });
    const cookie = sessionCookieFrom(login.setCookie);

    const created = await request("POST", `${APP_LOGIN_PATH_PREFIX}/users`, {
      cookie,
      body: { username: "alice", password: "secret1" },
    });
    expect(created.status).toBe(201);
    expect(created.json).toEqual({ username: "alice" });

    const listed = await request("GET", `${APP_LOGIN_PATH_PREFIX}/users`, {
      cookie,
    });
    expect(listed.status).toBe(200);
    expect(listed.json).toEqual({
      users: [{ username: "alice" }, { username: DEFAULT_APP_LOGIN_USERNAME }],
    });
  });

  it("deletes a user but refuses to delete the last one", async () => {
    const login = await request("POST", `${APP_LOGIN_PATH_PREFIX}/login`, {
      body: {
        username: DEFAULT_APP_LOGIN_USERNAME,
        password: DEFAULT_APP_LOGIN_PASSWORD,
      },
    });
    const cookie = sessionCookieFrom(login.setCookie);

    await request("POST", `${APP_LOGIN_PATH_PREFIX}/users`, {
      cookie,
      body: { username: "bob", password: "secret1" },
    });

    const deleted = await request(
      "DELETE",
      `${APP_LOGIN_PATH_PREFIX}/users/bob`,
      { cookie },
    );
    expect(deleted.status).toBe(200);

    const last = await request(
      "DELETE",
      `${APP_LOGIN_PATH_PREFIX}/users/${DEFAULT_APP_LOGIN_USERNAME}`,
      { cookie },
    );
    expect(last.status).toBe(400);
  });

  it("returns 404 for login routes when disabled", async () => {
    store = createAppLoginStore({ stateDir, enabled: false });
    handle = createAppLoginHandler(store);
    const status = await request("GET", `${APP_LOGIN_PATH_PREFIX}/status`);
    expect(status.json).toEqual({ enabled: false });

    const login = await request("POST", `${APP_LOGIN_PATH_PREFIX}/login`, {
      body: {
        username: DEFAULT_APP_LOGIN_USERNAME,
        password: DEFAULT_APP_LOGIN_PASSWORD,
      },
    });
    expect(login.status).toBe(404);
  });

  it("stores bcrypt hashes rather than plaintext passwords", async () => {
    const login = await request("POST", `${APP_LOGIN_PATH_PREFIX}/login`, {
      body: {
        username: DEFAULT_APP_LOGIN_USERNAME,
        password: DEFAULT_APP_LOGIN_PASSWORD,
      },
    });
    const cookie = sessionCookieFrom(login.setCookie);
    await request("POST", `${APP_LOGIN_PATH_PREFIX}/users`, {
      cookie,
      body: { username: "carol", password: "carol-pass" },
    });

    const raw = JSON.parse(await readFile(store.usersPath, "utf8")) as {
      users: Array<{ username: string; passwordHash: string }>;
    };
    const carol = raw.users.find((u) => u.username === "carol");
    expect(carol?.passwordHash).toMatch(/^\$2[aby]?\$/);
    expect(carol?.passwordHash).not.toContain("carol-pass");
    expect(await verifyAppLoginPassword("carol-pass", carol!.passwordHash)).toBe(
      true,
    );
  });
});
