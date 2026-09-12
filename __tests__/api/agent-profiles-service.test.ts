import { describe, expect, it, vi } from "vitest";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";

const mocks = vi.hoisted(() => ({
  kind: "local" as "local" | "cloud",
  list: vi.fn(),
  get: vi.fn(),
  save: vi.fn(),
  remove: vi.fn(),
  rename: vi.fn(),
  activate: vi.fn(),
  cloudList: vi.fn(),
  cloudGet: vi.fn(),
  cloudSave: vi.fn(),
  cloudRemove: vi.fn(),
  cloudRename: vi.fn(),
  cloudActivate: vi.fn(),
  Client: vi.fn(),
  clientOptions: vi.fn(),
}));

vi.mock("#/api/backend-registry/active-store", () => ({
  getActiveBackend: () => ({ backend: { kind: mocks.kind } }),
}));

vi.mock("#/api/agent-server-client-options", () => ({
  getAgentServerClientOptions: mocks.clientOptions,
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  AgentProfilesClient: mocks.Client,
}));

vi.mock("#/api/cloud/agent-profiles-service.api", () => ({
  listCloudAgentProfiles: mocks.cloudList,
  getCloudAgentProfile: mocks.cloudGet,
  saveCloudAgentProfile: mocks.cloudSave,
  deleteCloudAgentProfile: mocks.cloudRemove,
  renameCloudAgentProfile: mocks.cloudRename,
  activateCloudAgentProfile: mocks.cloudActivate,
}));

const setupMocks = () => {
  vi.clearAllMocks();
  mocks.kind = "local";
  mocks.clientOptions.mockReturnValue({ host: "http://localhost" });
  mocks.Client.mockImplementation(function AgentProfilesClientMock() {
    return {
      listAgentProfiles: mocks.list,
      getAgentProfile: mocks.get,
      saveAgentProfile: mocks.save,
      deleteAgentProfile: mocks.remove,
      renameAgentProfile: mocks.rename,
      activateAgentProfile: mocks.activate,
    };
  });
  for (const [mock, source] of [
    [mocks.list, "local-list"],
    [mocks.get, "local-get"],
    [mocks.save, "local-save"],
    [mocks.remove, "local-remove"],
    [mocks.rename, "local-rename"],
    [mocks.activate, "local-activate"],
    [mocks.cloudList, "cloud-list"],
    [mocks.cloudGet, "cloud-get"],
    [mocks.cloudSave, "cloud-save"],
    [mocks.cloudRemove, "cloud-remove"],
    [mocks.cloudRename, "cloud-rename"],
    [mocks.cloudActivate, "cloud-activate"],
  ] as const) {
    mock.mockResolvedValue({ source });
  }
};

describe("AgentProfilesService", () => {
  it("exports the seeded default profile name", async () => {
    vi.resetModules();
    const { WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME } = await import(
      "#/api/agent-profiles-service/agent-profiles-service.api"
    );
    expect(WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME).toBe("default");
  });

  it("routes every operation through a fresh local SDK client", async () => {
    setupMocks();
    const profile = { agent: "CodeActAgent" } as never;
    const results = await Promise.all([
      AgentProfilesService.listProfiles(),
      AgentProfilesService.getProfile("default", "encrypted"),
      AgentProfilesService.getProfile("plain"),
      AgentProfilesService.saveProfile("default", profile),
      AgentProfilesService.deleteProfile("old"),
      AgentProfilesService.renameProfile("old", "new"),
      AgentProfilesService.activateProfile("profile-id"),
    ]);

    expect(mocks.Client).toHaveBeenCalledTimes(7);
    expect(results).toEqual([
      { source: "local-list" },
      { source: "local-get" },
      { source: "local-get" },
      { source: "local-save" },
      { source: "local-remove" },
      { source: "local-rename" },
      { source: "local-activate" },
    ]);
    expect(mocks.get).toHaveBeenNthCalledWith(1, "default", {
      exposeSecrets: "encrypted",
    });
    expect(mocks.get).toHaveBeenNthCalledWith(2, "plain", {});
    expect(mocks.save).toHaveBeenCalledWith("default", profile);
    expect(mocks.remove).toHaveBeenCalledWith("old");
    expect(mocks.rename).toHaveBeenCalledWith("old", "new");
    expect(mocks.activate).toHaveBeenCalledWith("profile-id");
  });

  it("routes every operation to the cloud service", async () => {
    setupMocks();
    mocks.kind = "cloud";
    const profile = { agent: "CodeActAgent" } as never;
    const results = await Promise.all([
      AgentProfilesService.listProfiles(),
      AgentProfilesService.getProfile("default", "encrypted"),
      AgentProfilesService.saveProfile("default", profile),
      AgentProfilesService.deleteProfile("old"),
      AgentProfilesService.renameProfile("old", "new"),
      AgentProfilesService.activateProfile("profile-id"),
    ]);

    expect(mocks.cloudList).toHaveBeenCalledOnce();
    expect(results).toEqual([
      { source: "cloud-list" },
      { source: "cloud-get" },
      { source: "cloud-save" },
      { source: "cloud-remove" },
      { source: "cloud-rename" },
      { source: "cloud-activate" },
    ]);
    expect(mocks.cloudGet).toHaveBeenCalledWith("default");
    expect(mocks.cloudSave).toHaveBeenCalledWith("default", profile);
    expect(mocks.cloudRemove).toHaveBeenCalledWith("old");
    expect(mocks.cloudRename).toHaveBeenCalledWith("old", "new");
    expect(mocks.cloudActivate).toHaveBeenCalledWith("profile-id");
    expect(mocks.Client).not.toHaveBeenCalled();
  });
});
