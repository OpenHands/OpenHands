import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  agentProfileMayServeToolCatalog,
  agentProfileSupportsSecretRefs,
} from "#/api/agent-profiles-service/profile-field-support";
import type { AgentServerInfo } from "#/api/agent-server-compatibility";

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

describe("agentProfileMayServeToolCatalog", () => {
  it.each([null, undefined, {}, { capabilities: [] }])(
    "offers no picker when the catalog is unadvertised: %j",
    (info) => {
      expect(
        agentProfileMayServeToolCatalog("local", info as AgentServerInfo),
      ).toBe(false);
    },
  );

  it("offers the picker when the backend serves the catalog", () => {
    expect(
      agentProfileMayServeToolCatalog("local", {
        capabilities: ["tool_catalog_v1"],
      } as AgentServerInfo),
    ).toBe(true);
  });

  it("asks cloud, which advertises no capabilities", () => {
    expect(agentProfileMayServeToolCatalog("cloud", null)).toBe(true);
  });
});
