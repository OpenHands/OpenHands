import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  buildAutomationBackendEnv,
  buildLocalServiceRouteArgs,
} from "../../scripts/dev-static.mjs";

describe("dev-static", () => {
  const dirs: string[] = [];

  afterEach(() => {
    while (dirs.length > 0) {
      const dir = dirs.pop();
      if (dir) rmSync(dir, { recursive: true, force: true });
    }
  });

  function makeStateDir(): string {
    const dir = mkdtempSync(path.join(tmpdir(), "dev-static-state-"));
    dirs.push(dir);
    return dir;
  }

  it("uses the same session key for both agent-server and automation backend auth", () => {
    const env = buildAutomationBackendEnv(
      {
        agentServerPort: 18000,
        ingressPort: 8000,
        sessionApiKey: "shared-session-key",
        stateDir: makeStateDir(),
      },
      {},
    );

    // Both backends receive the same key value
    expect(env).toMatchObject({
      AUTOMATION_AGENT_SERVER_URL: "http://127.0.0.1:18000",
      AUTOMATION_AGENT_SERVER_API_KEY: "shared-session-key",
      AUTOMATION_LOCAL_API_KEY: "shared-session-key",
      AUTOMATION_POSTHOG_API_KEY:
        "phc_kBtz5nKmxVRRQ7HtPwr2QX9eMC5j65zE86QKocVNwb4U",
      AUTOMATION_POSTHOG_HOST: "https://us.i.posthog.com",
    });
  });

  // The Vite stack passes AUTOMATION_KV_SECRET; without it here the KV store
  // is simply off under `dev:static`, so automations that persist state
  // between runs behave differently depending on which launcher started them.
  it("passes the resolved KV secret through to the automation backend", () => {
    const env = buildAutomationBackendEnv(
      {
        agentServerPort: 18000,
        ingressPort: 8000,
        sessionApiKey: "shared-session-key",
        stateDir: makeStateDir(),
      },
      { AUTOMATION_KV_SECRET: "explicit-kv-secret" },
    );

    expect(env.AUTOMATION_KV_SECRET).toBe("explicit-kv-secret");
  });

  it("points every local proxy route at the IPv4 loopback", () => {
    // Both backends bind to `0.0.0.0`, which only accepts IPv4, so a
    // `localhost` target strands the proxy on ::1 on Windows.
    const args = buildLocalServiceRouteArgs({
      agentServerPort: 18000,
      autoBackendPort: 18001,
    });

    expect(args).toContain("/api/automation=http://127.0.0.1:18001");
    expect(args).toContain("/server_info=http://127.0.0.1:18000");
    expect(args.filter((arg: string) => arg.includes("localhost"))).toEqual([]);
  });
});
