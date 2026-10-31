// @vitest-environment node
import { createHash } from "node:crypto";
import { describe, expect, it, vi } from "vitest";

import {
  OAuthError,
  buildAuthorizeUrl,
  createPkce,
  createSignInAttempts,
  exchangeCode,
  readCallback,
  refreshGrant,
  renderReturnPage,
  revokeToken,
  signInWithDigitalOcean,
} from "../../scripts/mars-oauth.mjs";

const REDIRECT = "http://127.0.0.1:3000/mars/oauth/callback";

/** A token endpoint that answers like DigitalOcean's, recording each request. */
function fakeTokenEndpoint(
  reply: { status?: number; body: unknown } = {
    body: {
      access_token: "access-1",
      refresh_token: "refresh-1",
      token_type: "bearer",
      expires_in: 3600,
      info: {
        email: "me@example.com",
        team_uuid: "team-1",
        team_name: "My Team",
      },
    },
  },
) {
  const requests: URLSearchParams[] = [];
  const fetchImpl = vi.fn(async (_url: string, init: RequestInit) => {
    requests.push(new URLSearchParams(String(init.body)));
    return new Response(JSON.stringify(reply.body), {
      status: reply.status ?? 200,
      headers: { "Content-Type": "application/json" },
    });
  });
  return { fetchImpl: fetchImpl as unknown as typeof fetch, requests };
}

describe("buildAuthorizeUrl and createPkce", () => {
  it("asks for a code bound to a PKCE challenge, returning to the given address", () => {
    const { verifier, challenge } = createPkce();
    const url = new URL(
      buildAuthorizeUrl({
        clientId: "client-1",
        redirectUri: REDIRECT,
        state: "state-1",
        codeChallenge: challenge,
      }),
    );

    expect(url.searchParams.get("response_type")).toBe("code");
    expect(url.searchParams.get("code_challenge_method")).toBe("S256");
    expect(url.searchParams.get("code_challenge")).toBe(
      createHash("sha256").update(verifier).digest("base64url"),
    );
    expect(url.searchParams.get("redirect_uri")).toBe(REDIRECT);
    expect(url.searchParams.get("state")).toBe("state-1");
    // These permissions come from DigitalOcean's Harness Runtime API contract:
    // read lists, create provisions, operate resumes/connects, manage deletes.
    expect(new Set(url.searchParams.get("scope")?.split(" "))).toEqual(
      new Set([
        "agent_harness_session:read",
        "agent_harness_session:create",
        "agent_harness_session:operate",
        "agent_harness_session:manage",
      ]),
    );
    expect(url.toString()).not.toContain(verifier);
  });
});

describe("readCallback", () => {
  it("reads a full return address, or just its query string", () => {
    expect(readCallback(`${REDIRECT}?code=c1&state=s1`)).toEqual({
      code: "c1",
      state: "s1",
      error: null,
    });
    expect(readCallback("?code=c2&state=s2")).toMatchObject({
      code: "c2",
      state: "s2",
    });
  });

  it("reads the path and query the return route receives", () => {
    expect(readCallback("/mars/oauth/callback?code=c3&state=s3")).toEqual({
      code: "c3",
      state: "s3",
      error: null,
    });
  });

  it("surfaces the reason when the user declines", () => {
    expect(
      readCallback(
        "error=access_denied&error_description=The+user+denied+access&state=s1",
      ),
    ).toEqual({ code: null, state: "s1", error: "The user denied access" });
  });
});

describe("exchangeCode and refreshGrant", () => {
  it("trades the code and verifier for tokens, and names the team", async () => {
    const endpoint = fakeTokenEndpoint();

    const grant = await exchangeCode({
      clientId: "client-1",
      code: "code-1",
      verifier: "verifier-1",
      redirectUri: REDIRECT,
      fetchImpl: endpoint.fetchImpl,
    });

    expect(Object.fromEntries(endpoint.requests[0])).toEqual({
      grant_type: "authorization_code",
      code: "code-1",
      client_id: "client-1",
      redirect_uri: REDIRECT,
      code_verifier: "verifier-1",
    });
    expect(grant).toMatchObject({
      accessToken: "access-1",
      refreshToken: "refresh-1",
      account: { uuid: "team-1", name: "My Team", email: "me@example.com" },
    });
    const inAnHour = Date.now() + 3600_000;
    expect(Date.parse(grant.accessExpiresAt!)).toBeGreaterThan(inAnHour - 5000);
  });

  it("says when a refresh token is spent, so only a new sign-in helps", async () => {
    const endpoint = fakeTokenEndpoint({
      status: 400,
      body: { error: "invalid_grant", error_description: "Token is used" },
    });

    const refused = refreshGrant({
      clientId: "client-1",
      refreshToken: "refresh-1",
      fetchImpl: endpoint.fetchImpl,
    });

    await expect(refused).rejects.toBeInstanceOf(OAuthError);
    await expect(refused).rejects.toMatchObject({ isRevoked: true });
    expect(endpoint.requests[0].get("grant_type")).toBe("refresh_token");
  });

  it("treats DigitalOcean's answer to a reused refresh token as signed out, and a lost connection as not", async () => {
    // What cloud.digitalocean.com returned for a spent refresh token (20 Oct).
    const reused = fakeTokenEndpoint({
      status: 400,
      body: { error: "invalid_request" },
    });
    const offline = vi.fn(async () => {
      throw new TypeError("fetch failed");
    }) as unknown as typeof fetch;

    await expect(
      refreshGrant({
        clientId: "c",
        refreshToken: "spent",
        fetchImpl: reused.fetchImpl,
      }),
    ).rejects.toMatchObject({ isRevoked: true });
    await expect(
      refreshGrant({ clientId: "c", refreshToken: "r", fetchImpl: offline }),
    ).rejects.toMatchObject({ isRevoked: false });
  });
});

describe("revokeToken", () => {
  it("revokes an access token and says whether DigitalOcean confirmed it", async () => {
    const calls: RequestInit[] = [];
    const answer = (status: number) =>
      vi.fn(async (_url: string, init: RequestInit) => {
        calls.push(init);
        return new Response("{}", { status });
      }) as unknown as typeof fetch;

    expect(await revokeToken("access-1", answer(200))).toBe(true);
    expect(await revokeToken("access-1", answer(401))).toBe(false);
    expect(calls[0].headers).toMatchObject({
      Authorization: "Bearer access-1",
    });
    expect(new URLSearchParams(String(calls[0].body)).get("token")).toBe(
      "access-1",
    );
  });
});

describe("createSignInAttempts", () => {
  it("lets a return finish its sign-in once; a replay finds nothing", () => {
    const attempts = createSignInAttempts();
    const { attemptId, authorizeUrl } = attempts.begin({
      clientId: "client-1",
      redirectUri: REDIRECT,
    });
    const state = new URL(authorizeUrl).searchParams.get("state");

    expect(attempts.claim(state, attemptId)).toMatchObject({ id: attemptId });
    expect(attempts.claim(state, attemptId)).toBeNull();
  });

  it("ignores a return for another attempt, a guessed state, or an expired one", () => {
    let now = 0;
    const attempts = createSignInAttempts({ ttlMs: 1000, now: () => now });
    const first = attempts.begin({ clientId: "c", redirectUri: REDIRECT });
    const second = attempts.begin({ clientId: "c", redirectUri: REDIRECT });
    const firstState = new URL(first.authorizeUrl).searchParams.get("state");

    expect(attempts.claim(firstState, second.attemptId)).toBeNull();
    expect(attempts.claim("guessed")).toBeNull();
    now = 2000;
    expect(attempts.claim(firstState, first.attemptId)).toBeNull();
  });

  it("wakes a waiting page when the sign-in finishes, and reports failures", async () => {
    const attempts = createSignInAttempts();
    const ok = attempts.begin({ clientId: "c", redirectUri: REDIRECT });
    const bad = attempts.begin({ clientId: "c", redirectUri: REDIRECT });

    const waiting = attempts.wait(ok.attemptId, 10_000);
    attempts.succeed(ok.attemptId);
    attempts.fail(bad.attemptId, "Wrong account");

    await expect(waiting).resolves.toEqual({ status: "done", message: null });
    await expect(attempts.wait(bad.attemptId)).resolves.toEqual({
      status: "failed",
      message: "Wrong account",
    });
    await expect(attempts.wait("unknown")).resolves.toMatchObject({
      status: "failed",
    });
  });

  it("answers pending when nothing happened in time, and forgets a cancelled one", async () => {
    const attempts = createSignInAttempts();
    const { attemptId, authorizeUrl } = attempts.begin({
      clientId: "c",
      redirectUri: REDIRECT,
    });

    await expect(attempts.wait(attemptId, 5)).resolves.toEqual({
      status: "pending",
    });
    attempts.cancel(attemptId);
    expect(
      attempts.claim(new URL(authorizeUrl).searchParams.get("state")),
    ).toBeNull();
  });

  it("finishes a return claimed just before its sign-in expired", async () => {
    let now = 0;
    const attempts = createSignInAttempts({ ttlMs: 1000, now: () => now });
    const { attemptId, authorizeUrl } = attempts.begin({
      clientId: "c",
      redirectUri: REDIRECT,
    });
    now = 990;
    attempts.claim(new URL(authorizeUrl).searchParams.get("state"));
    const waiting = attempts.wait(attemptId, 10_000);

    // The exchange is still out when the ten minutes pass.
    now = 2000;
    attempts.begin({ clientId: "c", redirectUri: REDIRECT });
    attempts.succeed(attemptId);

    await expect(waiting).resolves.toEqual({ status: "done", message: null });
  });

  it("tells a waiting page its sign-in expired", async () => {
    let now = 0;
    const attempts = createSignInAttempts({ ttlMs: 1000, now: () => now });
    const { attemptId } = attempts.begin({
      clientId: "c",
      redirectUri: REDIRECT,
    });
    const waiting = attempts.wait(attemptId, 10_000);

    now = 2000;
    attempts.begin({ clientId: "c", redirectUri: REDIRECT });

    await expect(waiting).resolves.toEqual({
      status: "failed",
      message: "This sign-in expired. Start again.",
    });
  });
});

describe("renderReturnPage", () => {
  it("names the team and nothing else, escaped", () => {
    const page = renderReturnPage({ ok: true, teamName: "<b>Team</b>" });

    expect(page).toContain("&lt;b&gt;Team&lt;/b&gt;");
    expect(page).not.toContain("<b>Team</b>");
  });

  it("takes the spent code out of the address bar after a sign-in, and leaves a failed one's to paste", () => {
    const runPage = (ok: boolean) => {
      // The page's own script, run against a browser at the return address.
      const script = /<script>([\s\S]*?)<\/script>/.exec(
        renderReturnPage({ ok }),
      )![1];
      const history = { replaceState: vi.fn() };
      const location = { pathname: new URL(REDIRECT).pathname };
      new Function("history", "location", "setTimeout", script)(
        history,
        location,
        vi.fn(),
      );
      return history.replaceState;
    };

    expect(runPage(true)).toHaveBeenCalledWith(
      null,
      "",
      "/mars/oauth/callback",
    );
    expect(runPage(false)).not.toHaveBeenCalled();
  });
});

describe("signInWithDigitalOcean (desktop)", () => {
  it("returns to a loopback listener, ignores a forged return, and trades the real code", async () => {
    const endpoint = fakeTokenEndpoint();
    let authorizeUrl = "";
    const opened = new Promise<void>((resolve) => {
      void signInWithDigitalOcean({
        clientId: "client-1",
        fetchImpl: endpoint.fetchImpl,
        openExternal: async (url) => {
          authorizeUrl = url;
          resolve();
        },
      }).then((grant) => results.push(grant));
    });
    const results: unknown[] = [];
    await opened;
    const url = new URL(authorizeUrl);
    const redirect = new URL(url.searchParams.get("redirect_uri")!);
    const state = url.searchParams.get("state");

    expect(redirect.hostname).toBe("127.0.0.1");
    const forged = await fetch(`${redirect}?code=evil&state=guessed`);
    expect(forged.status).toBe(400);

    const real = await fetch(`${redirect}?code=code-1&state=${state}`);
    expect(real.status).toBe(200);
    expect(await real.text()).toContain("My Team");
    await vi.waitFor(() => expect(results).toHaveLength(1));
    expect(results[0]).toMatchObject({ accessToken: "access-1" });
    expect(endpoint.requests).toHaveLength(1);
    expect(endpoint.requests[0].get("code")).toBe("code-1");
    expect(endpoint.requests[0].get("redirect_uri")).toBe(redirect.toString());
  });

  it("ignores a declined return for another sign-in, and stops on its own", async () => {
    let forgedStatus: number | undefined;
    const signingIn = signInWithDigitalOcean({
      clientId: "client-1",
      fetchImpl: fakeTokenEndpoint().fetchImpl,
      openExternal: async (authorizeUrl) => {
        const url = new URL(authorizeUrl);
        const redirect = url.searchParams.get("redirect_uri");
        const state = url.searchParams.get("state");
        // A page that can reach the listener but doesn't know the state.
        forgedStatus = (
          await fetch(`${redirect}?error=access_denied&state=guessed`)
        ).status;
        await fetch(
          `${redirect}?error=access_denied&error_description=The+user+denied+access&state=${state}`,
        );
      },
    });

    await expect(signingIn).rejects.toThrow("The user denied access");
    expect(forgedStatus).toBe(400);
  });

  it("explains that sign-in is not set up when no client is configured", async () => {
    await expect(
      signInWithDigitalOcean({ clientId: "", openExternal: async () => {} }),
    ).rejects.toThrow(/personal access token/);
  });
});
