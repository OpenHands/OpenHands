import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  MIN_AGENT_SERVER_VERSION_FOR_PROFILE_SWITCH_LLM_TOOL,
  MIN_AGENT_SERVER_VERSION_FOR_PROFILE_TOOLS,
  agentProfileSupportsSwitchLlmTool,
  agentProfileSupportsTools,
} from "#/api/agent-profiles-service/profile-field-support";

const mockGetCachedAgentServerVersion = vi.fn<() => string | null>();

// Only the version *lookup* is mocked — the comparison stays the real one so
// these cases exercise the same parser the boot-time compatibility check uses.
vi.mock("#/api/agent-server-compatibility", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/api/agent-server-compatibility")>();
  return {
    ...actual,
    getCachedAgentServerVersion: () => mockGetCachedAgentServerVersion(),
  };
});

describe("agentProfileSupportsSwitchLlmTool", () => {
  beforeEach(() => {
    mockGetCachedAgentServerVersion.mockReset();
  });

  it("pins the gate to the release that added the profile field", () => {
    expect(MIN_AGENT_SERVER_VERSION_FOR_PROFILE_SWITCH_LLM_TOOL).toBe("1.31.0");
  });

  it.each(["1.29.0", "1.29.3", "1.30.0"])(
    "reports no support on %s, where agent profiles exist but the field does not",
    (version) => {
      mockGetCachedAgentServerVersion.mockReturnValue(version);
      expect(agentProfileSupportsSwitchLlmTool()).toBe(false);
    },
  );

  it.each(["1.31.0", "1.31.2", "1.36.1", "2.0.0"])(
    "reports support on %s",
    (version) => {
      mockGetCachedAgentServerVersion.mockReturnValue(version);
      expect(agentProfileSupportsSwitchLlmTool()).toBe(true);
    },
  );

  it("treats a prerelease of the gating version as older", () => {
    mockGetCachedAgentServerVersion.mockReturnValue("1.31.0-rc.1");
    expect(agentProfileSupportsSwitchLlmTool()).toBe(false);
  });

  it("assumes support when no version is cached (cloud backends)", () => {
    mockGetCachedAgentServerVersion.mockReturnValue(null);
    expect(agentProfileSupportsSwitchLlmTool()).toBe(true);
  });

  it("assumes support when the reported version does not parse", () => {
    mockGetCachedAgentServerVersion.mockReturnValue("main");
    expect(agentProfileSupportsSwitchLlmTool()).toBe(true);
  });
});

describe("agentProfileSupportsTools", () => {
  beforeEach(() => {
    mockGetCachedAgentServerVersion.mockReset();
  });

  it("pins the gate to the release that added the profile field", () => {
    expect(MIN_AGENT_SERVER_VERSION_FOR_PROFILE_TOOLS).toBe("1.31.2");
  });

  it.each(["1.29.0", "1.31.0", "1.31.1"])(
    "reports no support on %s, where a posted `tools` key would 422 the save",
    (version) => {
      mockGetCachedAgentServerVersion.mockReturnValue(version);
      expect(agentProfileSupportsTools()).toBe(false);
    },
  );

  it.each(["1.31.2", "1.36.1", "2.0.0"])("reports support on %s", (version) => {
    mockGetCachedAgentServerVersion.mockReturnValue(version);
    expect(agentProfileSupportsTools()).toBe(true);
  });

  it("assumes support when no version is cached (cloud backends)", () => {
    mockGetCachedAgentServerVersion.mockReturnValue(null);
    expect(agentProfileSupportsTools()).toBe(true);
  });
});
