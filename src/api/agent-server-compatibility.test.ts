import { describe, expect, it } from "vitest";
import {
  AgentServerUnknownVersionError,
  AgentServerUnsupportedVersionError,
  assertAgentServerVersionIsSupported,
  getDisplayAgentServerSdkVersion,
  getDisplayAgentServerVersion,
  type AgentServerInfo,
} from "./agent-server-compatibility";

const serverInfo = (version?: string, sdkVersion?: string): AgentServerInfo =>
  ({ version, sdk_version: sdkVersion }) as AgentServerInfo;

describe("agent-server version compatibility", () => {
  it("classifies missing and unknown versions separately from old versions", () => {
    for (const version of [undefined, "", "unknown", " UNKNOWN "]) {
      expect(() =>
        assertAgentServerVersionIsSupported(serverInfo(version)),
      ).toThrow(AgentServerUnknownVersionError);
    }
  });

  it("classifies malformed versions separately from old versions", () => {
    expect(() =>
      assertAgentServerVersionIsSupported(serverInfo("dev-build")),
    ).toThrow(AgentServerUnknownVersionError);
  });

  it("keeps valid but too-old versions on the unsupported-version path", () => {
    expect(() =>
      assertAgentServerVersionIsSupported(serverInfo("0.0.1")),
    ).toThrow(AgentServerUnsupportedVersionError);
  });

  it("does not render unknown or malformed versions as backend badges", () => {
    expect(getDisplayAgentServerVersion(serverInfo("unknown"))).toBeNull();
    expect(getDisplayAgentServerVersion(serverInfo("dev-build"))).toBeNull();
    expect(getDisplayAgentServerVersion(serverInfo("1.28.0"))).toBe("1.28.0");
  });

  it("uses sdk_version as the agent-server version fallback", () => {
    expect(getDisplayAgentServerVersion(serverInfo(undefined, "1.36.1"))).toBe(
      "1.36.1",
    );
    expect(() =>
      assertAgentServerVersionIsSupported(serverInfo(undefined, "1.36.1")),
    ).not.toThrow();
  });

  it("reports the dedicated agent-server SDK version when present", () => {
    expect(
      getDisplayAgentServerSdkVersion(serverInfo("1.36.0", "1.36.1")),
    ).toBe("1.36.1");
    expect(getDisplayAgentServerSdkVersion(serverInfo("1.36.0"))).toBe(
      "1.36.0",
    );
  });
});
