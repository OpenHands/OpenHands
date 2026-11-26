import { beforeEach, describe, expect, it, vi } from "vitest";
import type {
  Backend,
  ResolvedActiveBackend,
} from "#/api/backend-registry/types";
import RunningVersionsService from "#/api/running-versions-service";

const { callCloudProxy, getServerInfo, getAgentServerClientOptions } =
  vi.hoisted(() => ({
    callCloudProxy: vi.fn(),
    getServerInfo: vi.fn(),
    getAgentServerClientOptions: vi.fn(),
  }));

vi.mock("#/api/cloud/proxy", () => ({ callCloudProxy }));
vi.mock("#/api/agent-server-client-options", () => ({
  getAgentServerClientOptions,
}));
vi.mock("@openhands/typescript-client/clients", () => ({
  ServerClient: class {
    getServerInfo = getServerInfo;
  },
}));

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};
const cloudBackend: Backend = {
  id: "cloud",
  name: "Cloud",
  host: "https://app.example.test",
  apiKey: "cloud-key",
  kind: "cloud",
};
const localActive: ResolvedActiveBackend = {
  backend: localBackend,
  orgId: null,
};
const cloudActive: ResolvedActiveBackend = {
  backend: cloudBackend,
  orgId: "org-1",
};

describe("RunningVersionsService.getBackendVersions", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getAgentServerClientOptions.mockReturnValue({ host: localBackend.host });
  });

  it("reads the live local Agent Server package versions through the typed client", async () => {
    getServerInfo.mockResolvedValueOnce({
      version: "1.62.0",
      sdk_version: "1.62.1",
      tools_version: "1.62.2",
      workspace_version: "1.62.3",
    });

    await expect(
      RunningVersionsService.getBackendVersions(localActive),
    ).resolves.toEqual({
      serviceVersion: "1.62.0",
      sdkVersion: "1.62.1",
      toolsVersion: "1.62.2",
      workspaceVersion: "1.62.3",
    });

    expect(getAgentServerClientOptions).toHaveBeenCalledOnce();
    expect(getServerInfo).toHaveBeenCalledOnce();
    expect(callCloudProxy).not.toHaveBeenCalled();
  });

  it("reads the Cloud app and SDK versions from its server-info endpoint", async () => {
    callCloudProxy.mockResolvedValueOnce({
      app_version: "1.94.0",
      sdk_version: "1.62.0",
    });

    await expect(
      RunningVersionsService.getBackendVersions(cloudActive),
    ).resolves.toEqual({
      serviceVersion: "1.94.0",
      sdkVersion: "1.62.0",
      toolsVersion: null,
      workspaceVersion: null,
    });

    expect(callCloudProxy).toHaveBeenCalledWith(
      expect.objectContaining({
        backend: cloudBackend,
        method: "GET",
        path: "/server_info",
        headers: expect.objectContaining({ "X-Org-Id": "org-1" }),
        orgId: "org-1",
        timeoutSeconds: 5,
      }),
    );
    expect(getServerInfo).not.toHaveBeenCalled();
  });

  it("normalizes missing server package versions as unavailable", async () => {
    getServerInfo.mockResolvedValueOnce({ version: " " });

    await expect(
      RunningVersionsService.getBackendVersions(localActive),
    ).resolves.toBeNull();
  });

  it("treats an unavailable backend as a non-fatal missing version result", async () => {
    getServerInfo.mockRejectedValueOnce(new Error("offline"));

    await expect(
      RunningVersionsService.getBackendVersions(localActive),
    ).resolves.toBeNull();
  });
});
