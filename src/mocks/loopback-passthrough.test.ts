import { describe, expect, it, vi } from "vitest";
import { isLoopbackUrl } from "#/mocks/loopback-passthrough";

describe("isLoopbackUrl", () => {
  it("accepts loopback hosts in every spelling", () => {
    for (const url of [
      "http://localhost:8000/api/conversations",
      "http://127.0.0.1:5654/",
      "http://0.0.0.0:3000/",
      "http://[::1]:8000/api/conversations",
    ]) {
      expect(isLoopbackUrl(url), url).toBe(true);
    }
  });

  it("rejects a remote host that merely mentions a loopback address", () => {
    for (const url of [
      "https://localhost.example.com/",
      "https://notlocalhost.test/",
      "https://api.example.com/localhost/127.0.0.1",
      "https://registry.npmjs.org/@openhands/agent-canvas/latest",
    ]) {
      expect(isLoopbackUrl(url), url).toBe(false);
    }
  });

  it("does not treat an unparsable URL as local", () => {
    expect(isLoopbackUrl("not a url")).toBe(false);
  });
});

describe("global unhandled-request policy", () => {
  it("stops an unhandled remote request and reports the method and URL", async () => {
    const reported: string[] = [];
    const capture = (...args: unknown[]) => {
      reported.push(args.map(String).join(" "));
    };
    const consoleError = vi.spyOn(console, "error").mockImplementation(capture);
    const consoleWarn = vi.spyOn(console, "warn").mockImplementation(capture);

    const url = "https://unhandled-remote.test/v1/models";

    try {
      const error = await fetch(url).then(
        () => null,
        (reason: unknown) => reason as Error,
      );

      // A request that escaped to the network would reject with a transport error
      // (`fetch failed`, DNS) or, worse, resolve. Matching MSW's own unhandled
      // request error is what makes this meaningful for an unresolvable host.
      expect(error).not.toBeNull();
      expect(error?.message).toContain("Cannot bypass a request");
      expect(error?.message).not.toContain("fetch failed");

      // ...and the human-readable report names the request, so a failing test
      // points straight at the handler that is missing.
      const report = reported.join("\n");
      expect(report).toContain("without a matching request handler");
      expect(report).toContain("GET");
      expect(report).toContain(url);
    } finally {
      consoleError.mockRestore();
      consoleWarn.mockRestore();
    }
  });

  it("keeps the intentional loopback allowance in place", async () => {
    // Nothing listens on this port, so the request fails either way. What matters
    // is *why*: the policy would fail it with its own "Cannot bypass a request"
    // error, while the allowance lets it through to a transport error.
    const error = await fetch("http://127.0.0.1:1/health").then(
      () => null,
      (reason: unknown) => reason as Error,
    );

    expect(error).not.toBeNull();
    expect(error?.message).not.toContain("Cannot bypass a request");
  });
});
