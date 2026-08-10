/**
 * Client API for the internal app-login gate served by static-server/ingress.
 * These routes are NOT agent-server endpoints — they live on the canvas proxy.
 */

const APP_LOGIN_BASE = "/api/app-login";

export type AppLoginStatus = {
  enabled: boolean;
};

export type AppLoginSession = {
  authenticated: boolean;
  username?: string;
};

export type AppLoginUser = {
  username: string;
};

async function parseJson<T>(response: Response): Promise<T> {
  try {
    return (await response.json()) as T;
  } catch {
    throw new Error(`Unexpected response (${response.status})`);
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<
  | { ok: true; data: T; status: number }
  | { ok: false; error: string; status: number }
> {
  const response = await fetch(`${APP_LOGIN_BASE}${path}`, {
    credentials: "include",
    headers: {
      Accept: "application/json",
      ...(init.body ? { "Content-Type": "application/json" } : {}),
      ...init.headers,
    },
    ...init,
  });

  const data = await parseJson<T & { error?: string }>(response);
  if (!response.ok) {
    return {
      ok: false,
      status: response.status,
      error:
        typeof data?.error === "string" && data.error
          ? data.error
          : `Request failed (${response.status})`,
    };
  }
  return { ok: true, status: response.status, data };
}

export class AppLoginService {
  static async getStatus(): Promise<AppLoginStatus> {
    const result = await request<AppLoginStatus>("/status");
    if (!result.ok) {
      // If the proxy is an older build without the route, treat login as off.
      return { enabled: false };
    }
    return result.data;
  }

  static async getSession(): Promise<AppLoginSession> {
    const result = await request<AppLoginSession>("/me");
    if (!result.ok) {
      return { authenticated: false };
    }
    return result.data;
  }

  static async login(
    username: string,
    password: string,
  ): Promise<{ ok: true; username: string } | { ok: false; error: string }> {
    const result = await request<AppLoginSession>("/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    if (!result.ok) {
      return { ok: false, error: result.error };
    }
    return {
      ok: true,
      username: result.data.username ?? username,
    };
  }

  static async logout(): Promise<void> {
    await request("/logout", { method: "POST" });
  }

  static async listUsers(): Promise<AppLoginUser[]> {
    const result = await request<{ users: AppLoginUser[] }>("/users");
    if (!result.ok) {
      throw new Error(result.error);
    }
    return result.data.users;
  }

  static async createUser(
    username: string,
    password: string,
  ): Promise<AppLoginUser> {
    const result = await request<AppLoginUser>("/users", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    if (!result.ok) {
      throw new Error(result.error);
    }
    return result.data;
  }

  static async deleteUser(username: string): Promise<void> {
    const result = await request<{ deleted: string }>(
      `/users/${encodeURIComponent(username)}`,
      { method: "DELETE" },
    );
    if (!result.ok) {
      throw new Error(result.error);
    }
  }
}
