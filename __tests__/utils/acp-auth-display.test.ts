import { describe, it, expect } from "vitest";
import {
  resolveAcpAuthDisplay,
  type AcpAuthDisplay,
} from "#/utils/acp-auth-display";

// Pure decision matrix for the ACP auth banner. The host-login probe
// (`status`) is only trustworthy on a native backend; inside a container it
// can't run and returns "unknown". `credentialsConfigured` is the separate,
// backend-truthful signal (a secret exists in the active backend's store) that
// works on Docker/cloud too. See issue #1244.
describe("resolveAcpAuthDisplay", () => {
  it("reports a probe-confirmed login ('signed-in') whenever the probe authenticated, regardless of other signals", () => {
    expect(
      resolveAcpAuthDisplay({
        status: "authenticated",
        isChecking: false,
        credentialsConfigured: false,
      }),
    ).toBe("signed-in");
    expect(
      resolveAcpAuthDisplay({
        status: "authenticated",
        isChecking: true,
        credentialsConfigured: true,
      }),
    ).toBe("signed-in");
  });

  it("reports 'checking' while the first probe is in flight and no login is confirmed yet", () => {
    expect(
      resolveAcpAuthDisplay({
        status: "unknown",
        isChecking: true,
        credentialsConfigured: false,
      }),
    ).toBe("checking");
    expect(
      resolveAcpAuthDisplay({
        status: "unauthenticated",
        isChecking: true,
        credentialsConfigured: true,
      }),
    ).toBe("checking");
  });

  it("reports 'configured' when the probe can't confirm a login but a credential exists in the secret store (the Docker/cloud case)", () => {
    // "unknown" == couldn't tell (e.g. a container with no interactive CLI);
    // a stored secret is still a real signal the agent will authenticate.
    expect(
      resolveAcpAuthDisplay({
        status: "unknown",
        isChecking: false,
        credentialsConfigured: true,
      }),
    ).toBe("configured");
    expect(
      resolveAcpAuthDisplay({
        status: "unauthenticated",
        isChecking: false,
        credentialsConfigured: true,
      }),
    ).toBe("configured");
  });

  it("reports 'none' when there is no confirmed login and no stored credential", () => {
    expect(
      resolveAcpAuthDisplay({
        status: "unknown",
        isChecking: false,
        credentialsConfigured: false,
      }),
    ).toBe("none");
    expect(
      resolveAcpAuthDisplay({
        status: "unauthenticated",
        isChecking: false,
        credentialsConfigured: false,
      }),
    ).toBe("none");
  });

  it("never reports 'signed-in' from a stored credential alone — only the probe confirms a host login (honesty guard)", () => {
    const display: AcpAuthDisplay = resolveAcpAuthDisplay({
      status: "unknown",
      isChecking: false,
      credentialsConfigured: true,
    });
    expect(display).not.toBe("signed-in");
    expect(display).toBe("configured");
  });
});
