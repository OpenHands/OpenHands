import { describe, expect, it } from "vitest";

import {
  downgradeWorkspaceSessionCookie,
  rewriteWorkspaceSessionCookies,
} from "../../scripts/proxy-utils.mjs";

describe("downgradeWorkspaceSessionCookie", () => {
  const workspaceCookie =
    "oh_workspace_session_key=abc123; Path=/api/conversations; HttpOnly; Secure; SameSite=None; Partitioned";

  it("strips Secure and Partitioned and downgrades SameSite=None to Lax", () => {
    const result = downgradeWorkspaceSessionCookie(workspaceCookie);

    expect(result).toContain("oh_workspace_session_key=abc123");
    expect(result).toContain("Path=/api/conversations");
    expect(result).toContain("HttpOnly");
    expect(result.toLowerCase()).not.toContain("secure");
    expect(result.toLowerCase()).not.toContain("partitioned");
    expect(result).toContain("SameSite=Lax");
  });

  it("leaves non-workspace cookies unchanged", () => {
    const other = "other_cookie=xyz; Path=/; HttpOnly; Secure; SameSite=None";
    expect(downgradeWorkspaceSessionCookie(other)).toBe(other);
  });

  it("is a no-op when the workspace cookie has nothing to rewrite", () => {
    const alreadyLax =
      "oh_workspace_session_key=abc123; Path=/api/conversations; HttpOnly; SameSite=Lax";
    expect(downgradeWorkspaceSessionCookie(alreadyLax)).toBe(alreadyLax);
  });

  it("handles attribute casing", () => {
    const mixed =
      "oh_workspace_session_key=abc123; Path=/api/conversations; secure; samesite=none; partitioned";
    const result = downgradeWorkspaceSessionCookie(mixed);
    expect(result.toLowerCase()).not.toContain("secure");
    expect(result.toLowerCase()).not.toContain("partitioned");
    expect(result).toContain("SameSite=Lax");
  });
});

describe("rewriteWorkspaceSessionCookies", () => {
  it("rewrites the workspace cookie in an array and leaves others alone", () => {
    const proxyRes = {
      headers: {
        "set-cookie": [
          "oh_workspace_session_key=abc123; Path=/api/conversations; HttpOnly; Secure; SameSite=None; Partitioned",
          "other_cookie=xyz; Path=/; HttpOnly; Secure; SameSite=None",
        ],
      },
    };
    rewriteWorkspaceSessionCookies(proxyRes);

    const cookies = proxyRes.headers["set-cookie"] as string[];
    const workspace = cookies.find((c) =>
      c.startsWith("oh_workspace_session_key="),
    );
    const other = cookies.find((c) => c.startsWith("other_cookie="));

    expect(workspace).toBeDefined();
    expect(workspace!.toLowerCase()).not.toContain("secure");
    expect(workspace!.toLowerCase()).not.toContain("partitioned");
    expect(workspace).toContain("SameSite=Lax");

    expect(other).toBe(
      "other_cookie=xyz; Path=/; HttpOnly; Secure; SameSite=None",
    );
  });

  it("rewrites a single string header", () => {
    const proxyRes = {
      headers: {
        "set-cookie":
          "oh_workspace_session_key=abc123; Path=/api/conversations; HttpOnly; Secure; SameSite=None; Partitioned",
      },
    };
    rewriteWorkspaceSessionCookies(proxyRes);

    const cookie = proxyRes.headers["set-cookie"] as string;
    expect(cookie.toLowerCase()).not.toContain("secure");
    expect(cookie.toLowerCase()).not.toContain("partitioned");
    expect(cookie).toContain("SameSite=Lax");
  });

  it("is a no-op when no Set-Cookie header is present", () => {
    const proxyRes = { headers: {} };
    rewriteWorkspaceSessionCookies(proxyRes);
    expect(proxyRes.headers).toEqual({});
  });
});
