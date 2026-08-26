import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  fetchAgentServerVersion,
  testBackendConnection,
} from "#/api/backend-registry/backend-connection-service";
import { MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION } from "#/api/agent-server-compatibility";

const getServerInfoMock = vi.fn();
const getAgentServerClientOptionsMock = vi.fn((_overrides: unknown) => ({
  host: "http://localhost:8000",
  workingDir: "workspace/project",
}));

// The mock factories are hoisted above these consts, so both must reference
// them lazily from inside a function body rather than at factory-eval time.
vi.mock("@openhands/typescript-client/clients", () => ({
  ServerClient: vi.fn(function ServerClientMock() {
    return { getServerInfo: getServerInfoMock };
  }),
  SettingsClient: vi.fn(function SettingsClientMock() {
    return {};
  }),
}));

vi.mock("#/api/agent-server-client-options", () => ({
  getAgentServerClientOptions: (overrides: unknown) =>
    getAgentServerClientOptionsMock(overrides),
}));

const LOCAL_BACKEND = {
  host: "http://localhost:8000",
  apiKey: "sk-local",
  kind: "local" as const,
};

beforeEach(() => {
  getServerInfoMock.mockReset();
  getAgentServerClientOptionsMock.mockClear();
});

describe("testBackendConnection", () => {
  it("skips the probe for cloud backends", async () => {
    await expect(
      testBackendConnection({
        host: "https://app.all-hands.dev",
        apiKey: "sk-cloud",
        kind: "cloud",
      }),
    ).resolves.toEqual({ agentServerVersion: null });

    expect(getServerInfoMock).not.toHaveBeenCalled();
    expect(getAgentServerClientOptionsMock).not.toHaveBeenCalled();
  });

  it("returns the reported version for a supported local backend", async () => {
    getServerInfoMock.mockResolvedValue({
      version: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
    });

    await expect(testBackendConnection(LOCAL_BACKEND)).resolves.toEqual({
      agentServerVersion: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
    });
  });

  it("builds client options from the backend host, key and probe timeout", async () => {
    getServerInfoMock.mockResolvedValue({
      version: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
    });

    await testBackendConnection(LOCAL_BACKEND);

    expect(getAgentServerClientOptionsMock).toHaveBeenCalledWith({
      host: "http://localhost:8000",
      sessionApiKey: "sk-local",
      timeout: 5000,
    });
  });

  it("passes a null session key when the backend has no API key", async () => {
    getServerInfoMock.mockResolvedValue({
      version: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
    });

    await testBackendConnection({ ...LOCAL_BACKEND, apiKey: "" });

    expect(getAgentServerClientOptionsMock).toHaveBeenCalledWith(
      expect.objectContaining({ sessionApiKey: null }),
    );
  });

  it("rejects a local backend below the supported version floor", async () => {
    getServerInfoMock.mockResolvedValue({ version: "0.0.1" });

    await expect(testBackendConnection(LOCAL_BACKEND)).rejects.toThrow(
      /agent-server/i,
    );
  });

  it("rejects a local backend that does not report a usable version", async () => {
    getServerInfoMock.mockResolvedValue({ version: "unknown" });

    await expect(testBackendConnection(LOCAL_BACKEND)).rejects.toThrow(
      /version/i,
    );
  });

  it("propagates transport failures", async () => {
    getServerInfoMock.mockRejectedValue(new Error("Failed to fetch"));

    await expect(testBackendConnection(LOCAL_BACKEND)).rejects.toThrow(
      "Failed to fetch",
    );
  });
});

describe("fetchAgentServerVersion", () => {
  it("returns the display version", async () => {
    getServerInfoMock.mockResolvedValue({
      version: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
    });

    await expect(fetchAgentServerVersion(LOCAL_BACKEND)).resolves.toBe(
      MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
    );
  });

  it("does not enforce the version floor", async () => {
    // The status row reports whatever the server claims; only the submit-time
    // preflight rejects an unsupported version.
    getServerInfoMock.mockResolvedValue({ version: "0.0.1" });

    await expect(fetchAgentServerVersion(LOCAL_BACKEND)).resolves.toBe("0.0.1");
  });

  it("returns null when the reported version is unparseable", async () => {
    getServerInfoMock.mockResolvedValue({ version: "unknown" });

    await expect(fetchAgentServerVersion(LOCAL_BACKEND)).resolves.toBeNull();
  });

  it("reuses the same probe timeout as the connection test", async () => {
    getServerInfoMock.mockResolvedValue({
      version: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
    });

    await fetchAgentServerVersion(LOCAL_BACKEND);

    expect(getAgentServerClientOptionsMock).toHaveBeenCalledWith(
      expect.objectContaining({ timeout: 5000 }),
    );
  });
});
