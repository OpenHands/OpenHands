import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  agentProfileSupportsInstructions,
  agentProfileSupportsSecretRefs,
  agentProfileSupportsTools,
} from "#/api/agent-profiles-service/profile-field-support";

const mockServerInfo = vi.fn<() => { capabilities?: string[] } | null>();
const mockBackendKind = vi.fn<() => string>(() => "local");

vi.mock("#/api/backend-registry/active-store", () => ({
  getActiveBackend: () => ({ backend: { kind: mockBackendKind() } }),
}));

vi.mock("#/api/agent-server-compatibility", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/api/agent-server-compatibility")>();
  return {
    ...actual,
    getCachedAgentServerInfo: () => mockServerInfo(),
  };
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

describe("agentProfileSupportsInstructions", () => {
  it("offers instructions on a local backend", () => {
    mockBackendKind.mockReturnValue("local");
    expect(agentProfileSupportsInstructions()).toBe(true);
  });

  it("stays off on Cloud, whose launches do not apply them yet", () => {
    mockBackendKind.mockReturnValue("cloud");
    expect(agentProfileSupportsInstructions()).toBe(false);
  });
});

describe("agentProfileSupportsTools", () => {
  it("offers tools on a local backend", () => {
    mockBackendKind.mockReturnValue("local");
    expect(agentProfileSupportsTools()).toBe(true);
  });

  it("stays off on Cloud, which does not serve the tool catalog yet", () => {
    mockBackendKind.mockReturnValue("cloud");
    expect(agentProfileSupportsTools()).toBe(false);
  });
});
