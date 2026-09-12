import { describe, expect, it } from "vitest";

import { AUTH_HANDLERS } from "#/mocks/auth-handlers";
import { server } from "#/mocks/node";
import type { GitUser } from "#/types/git";

const API_ORIGIN = "http://auth.test";

function useAuthHandlers() {
  server.use(...AUTH_HANDLERS);
}

describe("mock authentication HTTP contracts", () => {
  it("reports that the cloud proxy is unavailable in mock mode", async () => {
    useAuthHandlers();

    const response = await fetch(`${API_ORIGIN}/api/cloud-proxy`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        host: "https://runtime.example.test",
        method: "GET",
        path: "/api/status",
      }),
    });

    expect(response.status).toBe(503);
    expect(response.headers.get("content-type")).toContain("application/json");
    await expect(response.json()).resolves.toEqual({
      error: "cloud proxy not available in mock mode",
    });
  });

  it("returns the mock Git user profile", async () => {
    useAuthHandlers();

    const response = await fetch(`${API_ORIGIN}/api/user/info`);
    const user = (await response.json()) as GitUser;

    expect(response.status).toBe(200);
    expect(user).toEqual({
      id: "1",
      login: "octocat",
      avatar_url: "https://avatars.githubusercontent.com/u/583231?v=4",
      company: "GitHub",
      email: "placeholder@placeholder.placeholder",
      name: "monalisa octocat",
    });
  });

  it("acknowledges authentication", async () => {
    useAuthHandlers();

    const response = await fetch(`${API_ORIGIN}/api/authenticate`, {
      method: "POST",
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      message: "Authenticated",
    });
  });

  it("acknowledges logout with a JSON null body", async () => {
    useAuthHandlers();

    const response = await fetch(`${API_ORIGIN}/api/logout`, {
      method: "POST",
    });

    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain("application/json");
    await expect(response.json()).resolves.toBeNull();
  });
});
