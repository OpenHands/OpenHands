// @vitest-environment node
import { createServer } from "node:net";
import { describe, expect, it, vi } from "vitest";

import {
  OAUTH_REDIRECT_PORT,
  OAUTH_REDIRECT_URI,
  buildAuthorizeUrl,
  parseImplicitGrant,
  signInWithDigitalOcean,
} from "../../scripts/mars-oauth.mjs";

const STATE = "state-abc";

describe("buildAuthorizeUrl", () => {
  it("requests the implicit grant against the registered redirect URI", () => {
    const url = new URL(
      buildAuthorizeUrl({ clientId: "client-1", state: STATE }),
    );

    // Authorization-code would require shipping a client_secret in the
    // desktop binary, which DigitalOcean documents no PKCE alternative for.
    expect(url.searchParams.get("response_type")).toBe("token");
    expect(url.searchParams.get("client_id")).toBe("client-1");
    expect(url.searchParams.get("state")).toBe(STATE);
    expect(url.searchParams.get("redirect_uri")).toBe(OAUTH_REDIRECT_URI);
  });
});

describe("parseImplicitGrant", () => {
  it("recovers the token from a relayed fragment", () => {
    const grant = parseImplicitGrant(
      `access_token=tok-1&token_type=bearer&expires_in=2592000&state=${STATE}`,
      STATE,
    );

    expect(grant.accessToken).toBe("tok-1");
    expect(Date.parse(grant.expiresAt ?? "")).toBeGreaterThan(Date.now());
  });

  it("rejects a callback whose state does not match the attempt", () => {
    expect(() =>
      parseImplicitGrant("access_token=tok-1&state=forged-state", STATE),
    ).toThrow(/state did not match/i);
  });

  it("treats a query-only callback with no fragment as a failure", () => {
    // Browsers never send the fragment to the server, so the bare loopback
    // request carries no token — without the relay page the flow would
    // silently hang here instead of reporting anything.
    expect(() => parseImplicitGrant("", STATE)).toThrow(
      /did not return an access token/i,
    );
  });

  it("surfaces the provider's error description when access is declined", () => {
    expect(() =>
      parseImplicitGrant(
        "error=access_denied&error_description=The+user+denied+access",
        STATE,
      ),
    ).toThrow("The user denied access");
  });
});

describe("signInWithDigitalOcean", () => {
  it("completes when the relay page posts the fragment back", async () => {
    const openExternal = vi.fn(async (authorizeUrl) => {
      const state = new URL(authorizeUrl).searchParams.get("state");

      // Stand in for the browser: fetch the callback page, then replay the
      // fragment to the relay route the way its inline script does.
      await fetch(OAUTH_REDIRECT_URI);
      await fetch(`http://127.0.0.1:${OAUTH_REDIRECT_PORT}/callback-token`, {
        method: "POST",
        body: `access_token=tok-2&token_type=bearer&expires_in=60&state=${state}`,
      });
    });

    const grant = await signInWithDigitalOcean({
      clientId: "client-1",
      openExternal,
    });

    expect(grant.accessToken).toBe("tok-2");
    expect(openExternal).toHaveBeenCalledOnce();
  });

  it("explains a busy redirect port and leaves no rejection behind", async () => {
    // Arrange
    const blocker = createServer();
    await new Promise<void>((resolve) =>
      blocker.listen(OAUTH_REDIRECT_PORT, "127.0.0.1", resolve),
    );
    const unhandled = vi.fn();
    process.on("unhandledRejection", unhandled);
    const openExternal = vi.fn();

    try {
      // Act
      const attempt = signInWithDigitalOcean({
        clientId: "client-1",
        openExternal,
        timeoutMs: 20,
      });

      // Assert
      await expect(attempt).rejects.toThrow(
        `Port ${OAUTH_REDIRECT_PORT} is already in use`,
      );
      await new Promise((resolve) => setTimeout(resolve, 60));
      expect(unhandled).not.toHaveBeenCalled();
      expect(openExternal).not.toHaveBeenCalled();
    } finally {
      process.off("unhandledRejection", unhandled);
      await new Promise((resolve) => blocker.close(resolve));
    }
  });

  it("explains that a token is the way forward when no OAuth client is built in", async () => {
    await expect(
      signInWithDigitalOcean({ clientId: null, openExternal: vi.fn() }),
    ).rejects.toThrow(/personal access token/i);
  });
});
