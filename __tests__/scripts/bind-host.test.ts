import { describe, expect, it, vi } from "vitest";

import {
  DEFAULT_BIND_HOST,
  applySessionKeyPolicy,
  bindHostArgs,
  isLoopbackBind,
  resolveBindHost,
} from "../../scripts/bind-host.mjs";

describe("bind-host", () => {
  it("defaults to IPv4 loopback", () => {
    expect(DEFAULT_BIND_HOST).toBe("127.0.0.1");
    expect(resolveBindHost({})).toBe("127.0.0.1");
  });

  it("prefers --host over OH_BIND_HOST", () => {
    expect(
      resolveBindHost({ flag: "0.0.0.0", env: "::" }),
    ).toBe("0.0.0.0");
  });

  it("treats :: and 0.0.0.0 as non-loopback", () => {
    expect(isLoopbackBind("::")).toBe(false);
    expect(isLoopbackBind("0.0.0.0")).toBe(false);
    expect(isLoopbackBind("127.0.0.1")).toBe(true);
    expect(isLoopbackBind("::1")).toBe(true);
    expect(isLoopbackBind("[::1]")).toBe(true);
    expect(isLoopbackBind("localhost")).toBe(true);
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
});
