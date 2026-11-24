import { describe, expect, it, vi } from "vitest";

import {
  DEFAULT_BIND_HOST,
  applySessionKeyPolicy,
  bindHostArgs,
  isCrossOriginHttpRequest,
  isForeignLoopbackRequest,
  isLoopbackBind,
  isLoopbackHostname,
  parseHttpHostHeader,
  parseHttpOriginHeader,
  resolveBindHost,
} from "../../scripts/bind-host.mjs";

describe("bind-host", () => {
  it("defaults to IPv4 loopback", () => {
    expect(DEFAULT_BIND_HOST).toBe("127.0.0.1");
    expect(resolveBindHost({})).toBe("127.0.0.1");
  });

  it("prefers --host over OH_BIND_HOST", () => {
    expect(resolveBindHost({ flag: "0.0.0.0", env: "::" })).toBe("0.0.0.0");
  });

  it("treats :: and 0.0.0.0 as non-loopback", () => {
    expect(isLoopbackBind("::")).toBe(false);
    expect(isLoopbackBind("0.0.0.0")).toBe(false);
    expect(isLoopbackBind("127.0.0.1")).toBe(true);
    expect(isLoopbackBind("::1")).toBe(true);
    expect(isLoopbackBind("[::1]")).toBe(true);
    expect(isLoopbackBind("localhost")).toBe(true);
  });

  it("recognizes the full 127.0.0.0/8 loopback range (RFC 3330)", () => {
    expect(isLoopbackBind("127.0.0.2")).toBe(true);
    expect(isLoopbackBind("127.0.0.3")).toBe(true);
    expect(isLoopbackBind("127.1.2.3")).toBe(true);
    expect(isLoopbackBind("127.255.255.255")).toBe(true);
  });

  it("strips the session key when bound off-loopback", () => {
    const warn = vi.fn();
    const policy = applySessionKeyPolicy({
      host: "::",
      sessionApiKey: "secret-key",
      warn,
    });
    expect(policy.sessionApiKey).toBeNull();
    expect(policy.authRequired).toBe(true);
    expect(policy.strippedSessionKey).toBe(true);
    expect(warn).toHaveBeenCalled();
  });

  it("keeps the session key on loopback", () => {
    const policy = applySessionKeyPolicy({
      host: "127.0.0.1",
      sessionApiKey: "secret-key",
      warn: () => {},
    });
    expect(policy.sessionApiKey).toBe("secret-key");
    expect(policy.authRequired).toBe(false);
    expect(policy.strippedSessionKey).toBe(false);
  });

  it("keeps the session key on non-loopback when explicitly allowed", () => {
    const policy = applySessionKeyPolicy({
      host: "::",
      sessionApiKey: "secret-key",
      allowLanSessionKey: true,
      warn: () => {},
    });
    expect(policy.sessionApiKey).toBe("secret-key");
    expect(policy.strippedSessionKey).toBe(false);
  });

  it("emits --host for child processes", () => {
    expect(bindHostArgs("127.0.0.1")).toEqual(["--host", "127.0.0.1"]);
  });

  it("parses Host headers strictly without allowing userinfo or delimiters", () => {
    expect(parseHttpHostHeader("LOCALHOST:8000")).toEqual({
      host: "localhost:8000",
      hostname: "localhost",
      port: "8000",
    });
    expect(parseHttpHostHeader("[::1]:8000")).toEqual({
      host: "[::1]:8000",
      hostname: "::1",
      port: "8000",
    });
    expect(parseHttpHostHeader("evil.example@127.0.0.1:8000")).toBeNull();
    expect(parseHttpHostHeader("127.0.0.1:8000, evil.example")).toBeNull();
    expect(parseHttpHostHeader("127.999.0.1:8000")).toBeNull();
    expect(parseHttpHostHeader("")).toBeNull();
    expect(parseHttpHostHeader(undefined)).toBeNull();
  });

  it("validates loopback hostnames and Origin headers", () => {
    expect(isLoopbackHostname("127.0.0.1")).toBe(true);
    expect(isLoopbackHostname("::1")).toBe(true);
    expect(isLoopbackHostname("localhost")).toBe(true);
    expect(isLoopbackHostname("canvas.localhost")).toBe(true);
    expect(isLoopbackHostname(".localhost")).toBe(false);
    expect(isLoopbackHostname("foo..localhost")).toBe(false);
    expect(isLoopbackHostname("localhost.evil.example")).toBe(false);
    expect(isLoopbackHostname("127.999.0.1")).toBe(false);

    expect(parseHttpOriginHeader("http://LOCALHOST:3001")).toEqual({
      host: "localhost:3001",
      hostname: "localhost",
    });
    expect(parseHttpOriginHeader("null")).toBeNull();
    expect(parseHttpOriginHeader("file:///etc/passwd")).toBeNull();
    expect(parseHttpOriginHeader("http://user@localhost:8000")).toBeNull();
  });

  it("detects foreign Hosts, userinfo spoofs, and cross-origin WS/mutations on loopback", () => {
    const req = (headers: Record<string, string | undefined>, method = "GET") =>
      ({ headers, method }) as never;

    expect(
      isForeignLoopbackRequest(req({ host: "localhost:8000" }), "127.0.0.1"),
    ).toBe(false);
    expect(
      isForeignLoopbackRequest(req({ host: "sub.localhost:8000" }), "::1"),
    ).toBe(false);
    expect(
      isForeignLoopbackRequest(req({ host: "evil.example:8000" }), "127.0.0.1"),
    ).toBe(true);
    expect(
      isForeignLoopbackRequest(
        req({ host: "evil.example@127.0.0.1:8000" }),
        "127.0.0.1",
      ),
    ).toBe(true);
    expect(
      isForeignLoopbackRequest(req({ host: ".localhost:8000" }), "127.0.0.1"),
    ).toBe(true);
    expect(isForeignLoopbackRequest(req({}), "127.0.0.1")).toBe(true);

    // WebSocket upgrades and unsafe HTTP methods require a loopback Origin if sent
    expect(
      isForeignLoopbackRequest(
        req({ host: "127.0.0.1:8000", origin: "https://evil.example" }),
        "127.0.0.1",
        { upgrade: true },
      ),
    ).toBe(true);
    expect(
      isForeignLoopbackRequest(
        req({ host: "127.0.0.1:8000", origin: "null" }),
        "127.0.0.1",
        { upgrade: true },
      ),
    ).toBe(true);
    expect(
      isForeignLoopbackRequest(
        req({ host: "127.0.0.1:8000", origin: "https://evil.example" }, "POST"),
        "127.0.0.1",
      ),
    ).toBe(true);
    expect(
      isForeignLoopbackRequest(
        req(
          { host: "127.0.0.1:8000", origin: "http://localhost:3001" },
          "POST",
        ),
        "127.0.0.1",
      ),
    ).toBe(false);

    // Non-loopback binds remain permissive for reverse proxies / LAN
    expect(
      isForeignLoopbackRequest(
        req({
          host: "canvas.example.com",
          origin: "https://canvas.example.com",
        }),
        "::",
        { upgrade: true },
      ),
    ).toBe(false);

    // Shared cross-origin check rejects userinfo and honors case-insensitive match
    expect(
      isCrossOriginHttpRequest(
        req({ host: "LOCALHOST:8000", origin: "http://localhost:8000" }),
      ),
    ).toBe(false);
    expect(
      isCrossOriginHttpRequest(
        req({
          host: "evil.example@127.0.0.1:8000",
          origin: "http://127.0.0.1:8000",
        }),
      ),
    ).toBe(true);
  });
});
