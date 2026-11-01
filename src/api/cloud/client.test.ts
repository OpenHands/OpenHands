import { beforeEach, describe, expect, it, vi } from "vitest";

const {
  CloudClientMock,
  getAgentServerBaseUrlMock,
  getAgentServerHeadersMock,
} = vi.hoisted(() => ({
  CloudClientMock: vi.fn(function CloudClient(
    this: { proxy?: unknown },
    options: { proxy?: unknown },
  ) {
    this.proxy = options.proxy;
  }),
  getAgentServerBaseUrlMock: vi.fn(),
  getAgentServerHeadersMock: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  CloudClient: CloudClientMock,
}));

vi.mock("../agent-server-config", () => ({
  getAgentServerBaseUrl: getAgentServerBaseUrlMock,
  getAgentServerHeaders: getAgentServerHeadersMock,
}));

vi.mock("../backend-registry/active-store", () => ({
  getActiveBackend: () => ({
    backend: {
      id: "locked-cloud",
      name: "OpenHands Cloud",
      host: "https://cloud.example.test",
      apiKey: "",
      kind: "cloud",
      authMode: "cookie",
    },
    orgId: "org-1",
  }),
}));

import { createCloudClientForRuntime } from "./client";

describe("createCloudClientForRuntime", () => {
  beforeEach(() => {
    CloudClientMock.mockClear();
    getAgentServerBaseUrlMock.mockReset();
    getAgentServerHeadersMock.mockReset();
  });

  it("uses the cloud host as the runtime proxy in frontend-only SaaS", () => {
    getAgentServerBaseUrlMock.mockReturnValue(null);

    createCloudClientForRuntime();

    expect(CloudClientMock).toHaveBeenCalledWith(
      expect.objectContaining({
        host: "https://cloud.example.test",
        orgId: "org-1",
        proxy: { host: "https://cloud.example.test", headers: {} },
      }),
    );
  });

  it("keeps using the local Agent Server proxy when configured", () => {
    getAgentServerBaseUrlMock.mockReturnValue("http://localhost:8000");
    getAgentServerHeadersMock.mockReturnValue({
      "X-Session-API-Key": "session-key",
    });

    createCloudClientForRuntime();

    expect(CloudClientMock).toHaveBeenCalledWith(
      expect.objectContaining({
        proxy: {
          host: "http://localhost:8000",
          headers: { "X-Session-API-Key": "session-key" },
        },
      }),
    );
  });
});
