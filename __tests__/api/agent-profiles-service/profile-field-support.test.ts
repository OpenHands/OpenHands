import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  agentProfileSupportsSecretRefs,
  agentProfileSupportsToolCatalog,
} from "#/api/agent-profiles-service/profile-field-support";

const mockServerInfo = vi.fn<() => { capabilities?: string[] } | null>();
const mockBackendKind = vi.fn<() => string>(() => "local");

vi.mock("#/api/backend-registry/active-store", () => ({
  getActiveBackend: () => ({ backend: { kind: mockBackendKind() } }),
}));

vi.mock("#/api/agent-server-compatibility", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/api/agent-server-compatibility")>();
  return { ...actual, getCachedAgentServerInfo: () => mockServerInfo() };
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

describe("agentProfileSupportsToolCatalog", () => {
  beforeEach(() => {
    mockBackendKind.mockReturnValue("local");
  });

  it.each([null, {}, { capabilities: [] }])(
    "offers no picker when the catalog is unadvertised: %j",
    (info) => {
      mockServerInfo.mockReturnValue(info);
      expect(agentProfileSupportsToolCatalog()).toBe(false);
    },
  );

  it("offers the picker when the backend serves the catalog", () => {
    mockServerInfo.mockReturnValue({ capabilities: ["tool_catalog_v1"] });
    expect(agentProfileSupportsToolCatalog()).toBe(true);
  });

  it("offers no picker on cloud, which does not serve the catalog", () => {
    mockBackendKind.mockReturnValue("cloud");
    mockServerInfo.mockReturnValue({ capabilities: ["tool_catalog_v1"] });
    expect(agentProfileSupportsToolCatalog()).toBe(false);
  });
});
