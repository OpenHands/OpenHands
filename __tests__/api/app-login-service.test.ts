import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppLoginService } from "#/api/app-login-service";

describe("AppLoginService", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("treats a missing status endpoint as login disabled", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ error: "Not found" }), { status: 404 }),
    );

    await expect(AppLoginService.getStatus()).resolves.toEqual({
      enabled: false,
    });
  });

  it("posts login credentials with cookies included", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({ authenticated: true, username: "heimdallsec" }),
        { status: 200 },
      ),
    );

    await expect(
      AppLoginService.login("heimdallsec", "heimdallsec"),
    ).resolves.toEqual({ ok: true, username: "heimdallsec" });

    expect(fetchSpy).toHaveBeenCalledWith(
      "/api/app-login/login",
      expect.objectContaining({
        method: "POST",
        credentials: "include",
        body: JSON.stringify({
          username: "heimdallsec",
          password: "heimdallsec",
        }),
      }),
    );
  });
});
