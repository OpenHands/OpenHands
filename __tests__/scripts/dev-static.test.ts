import { describe, expect, it } from "vitest";

import {
  buildAutomationBackendEnv,
  buildLocalServiceRouteArgs,
  parseArgs,
  buildMarsWebEnv,
  buildMaritimeWebEnv,
} from "../../scripts/dev-static.mjs";
import { buildFrontendEnv } from "../../scripts/static-build.mjs";

describe("dev-static CLI", () => {
  it.each([
    [["--host", "0.0.0.0"], "0.0.0.0"],
    [["-H", "::"], "::"],
  ])("parses an explicit bind host", (argv, expected) => {
    expect(parseArgs(argv).host).toBe(expected);
  });
});

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

  it("keeps reusable frontend builds free of session credentials", () => {
    const env = buildFrontendEnv(
      { viteWorkingDir: "/tmp/workspace" },
      { VITE_SESSION_API_KEY: "inherited-secret" },
    );

    expect(env.VITE_SESSION_API_KEY).toBe("");
    expect(env.VITE_WORKING_DIR).toBe("/tmp/workspace");
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

describe("dev-static MARS web bridge env", () => {
  it("puts the bridge on the ingress behind the session key, and off the static child", () => {
    const config = { sessionApiKey: "stack-key" } as never;
    expect(buildMarsWebEnv("ingress", config)).toEqual({
      MARS_WEB_KEY: "stack-key",
    });
    expect(buildMarsWebEnv("static", config)).toEqual({ MARS_WEB: "0" });
  });

  it("tells the ingress bridge where to keep sign-ins and how to encrypt them", () => {
    const config = {
      sessionApiKey: "stack-key",
      stateDir: "/state",
      secretKey: "settings-key",
    } as never;
    expect(buildMarsWebEnv("ingress", config)).toEqual({
      MARS_WEB_KEY: "stack-key",
      MARS_STATE_DIR: "/state",
      OH_SECRET_KEY: "settings-key",
    });
  });

  it("mounts nothing when the stack has no session key", () => {
    expect(buildMarsWebEnv("ingress", {} as never)).toEqual({ MARS_WEB: "0" });
  });
});

describe("dev-static Maritime web bridge env", () => {
  it("places the Maritime bridge like the MARS one without opting in", () => {
    const config = {
      sessionApiKey: "stack-key",
      stateDir: "/state",
      secretKey: "settings-key",
    } as never;
    expect(buildMaritimeWebEnv("ingress", config)).toEqual({
      MARITIME_WEB_KEY: "stack-key",
      MARITIME_STATE_DIR: "/state",
      OH_SECRET_KEY: "settings-key",
    });
    expect(buildMaritimeWebEnv("static", config)).toEqual({
      MARITIME_WEB: "0",
    });
    expect(buildMaritimeWebEnv("ingress", {} as never)).toEqual({
      MARITIME_WEB: "0",
    });
  });
});
