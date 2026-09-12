import { describe, expect, it, vi } from "vitest";

vi.mock("../../config/defaults.json", () => ({
  default: {
    compatibility: {
      minimumAgentServer: "1.28.0-rc.2",
    },
  },
}));

import {
  AgentServerUnsupportedVersionError,
  assertAgentServerVersionIsSupported,
  MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
  type AgentServerInfo,
} from "./agent-server-compatibility";

const serverInfo = (version: string): AgentServerInfo =>
  ({ version }) as AgentServerInfo;

describe("agent-server prerelease minimum compatibility", () => {
  it("accepts the stable release for a prerelease minimum", () => {
    expect(MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION).toBe("1.28.0-rc.2");

    expect(() =>
      assertAgentServerVersionIsSupported(serverInfo("1.28.0")),
    ).not.toThrow();
  });

  it("compares prerelease identifiers at the configured version boundary", () => {
    expect(() =>
      assertAgentServerVersionIsSupported(serverInfo("1.28.0-rc.1")),
    ).toThrow(AgentServerUnsupportedVersionError);
    expect(() =>
      assertAgentServerVersionIsSupported(serverInfo("1.28.0-rc.2")),
    ).not.toThrow();
    expect(() =>
      assertAgentServerVersionIsSupported(serverInfo("1.28.0-rc.3")),
    ).not.toThrow();
  });
});
