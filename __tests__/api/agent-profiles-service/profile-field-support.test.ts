import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  MIN_AGENT_SERVER_VERSION_FOR_PROFILE_SWITCH_LLM_TOOL,
  agentProfileSupportsSecretRefs,
  agentProfileSupportsSwitchLlmTool,
} from "#/api/agent-profiles-service/profile-field-support";

const mockGetCachedAgentServerVersion = vi.fn<() => string | null>();
const mockServerInfo = vi.fn<() => { capabilities?: string[] } | null>();
const mockBackendKind = vi.fn<() => string>(() => "local");

vi.mock("#/api/backend-registry/active-store", () => ({
  getActiveBackend: () => ({ backend: { kind: mockBackendKind() } }),
}));

// Only the version *lookup* is mocked — the comparison stays the real one so
// these cases exercise the same parser the boot-time compatibility check uses.
vi.mock("#/api/agent-server-compatibility", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/api/agent-server-compatibility")>();
  return {
    ...actual,
    getCachedAgentServerVersion: () => mockGetCachedAgentServerVersion(),
    getCachedAgentServerInfo: () => mockServerInfo(),
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

describe("agentProfileSupportsSecretRefs", () => {
  beforeEach(() => {
    mockBackendKind.mockReturnValue("local");
  });

  it.each([null, {}, { capabilities: [] }])(
    "hides an unadvertised scope: %j",
    (info) => {
      mockServerInfo.mockReturnValue(info);
      expect(agentProfileSupportsSecretRefs()).toBe(false);
    },
  );

  it("accepts enforcement advertised by a local development build", () => {
    mockServerInfo.mockReturnValue({
      capabilities: ["profile_secret_scope_v1"],
    });
    expect(agentProfileSupportsSecretRefs()).toBe(true);
  });

  it("does not promise enforcement through the Cloud profile launch path", () => {
    mockBackendKind.mockReturnValue("cloud");
    mockServerInfo.mockReturnValue({
      capabilities: ["profile_secret_scope_v1"],
    });
    expect(agentProfileSupportsSecretRefs()).toBe(false);
  });
});
