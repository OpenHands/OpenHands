import { describe, expect, it } from "vitest";

import {
  buildAutomationBackendEnv,
  buildLocalServiceRouteArgs,
} from "../../scripts/dev-static.mjs";

describe("dev-static", () => {
  it("uses the same session key for both agent-server and automation backend auth", () => {
    const env = buildAutomationBackendEnv(
      {
        agentServerPort: 18000,
        ingressPort: 8000,
        sessionApiKey: "shared-session-key",
        stateDir: "/tmp/agent-canvas-state",
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
