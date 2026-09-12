import {
  ServerClient,
  SettingsClient,
} from "@openhands/typescript-client/clients";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import {
  AGENT_SERVER_UNKNOWN_VERSION_ERROR_CODE,
  AGENT_SERVER_UNSUPPORTED_VERSION_ERROR_CODE,
  AgentServerUnavailableError,
  AgentServerUnknownVersionError,
  AgentServerUnsupportedVersionError,
  clearCachedAgentServerInfo,
  isAgentServerAuthError,
  isAgentServerToolAvailable,
  isAgentServerUnavailableError,
  isAgentServerUnknownVersionError,
  isAgentServerUnsupportedVersionError,
  loadAgentServerInfo,
  MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
} from "#/api/agent-server-compatibility";

const { getServerInfoMock, getSettingsMock } = vi.hoisted(() => ({
  getServerInfoMock: vi.fn(),
  getSettingsMock: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  ServerClient: vi.fn(function ServerClientMock() {
    return {
      getServerInfo: getServerInfoMock,
    };
  }),
  SettingsClient: vi.fn(function SettingsClientMock() {
    return {
      getSettings: getSettingsMock,
    };
  }),
}));

const httpError = (status: number) =>
  Object.assign(new Error(`HTTP ${status}`), {
    name: "HttpError",
    status,
  });

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://localhost:9000",
  apiKey: "local-key",
  kind: "local",
};

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  getServerInfoMock.mockReset();
  getSettingsMock.mockReset();
  vi.mocked(ServerClient).mockClear();
  vi.mocked(SettingsClient).mockClear();
  getServerInfoMock.mockResolvedValue({
    version: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
  });
  getSettingsMock.mockResolvedValue({});
  clearCachedAgentServerInfo();
  delete (window as unknown as Record<string, unknown>)
    .__AGENT_CANVAS_AUTH_REQUIRED__;
});

afterEach(() => {
  vi.restoreAllMocks();
  window.localStorage.clear();
  __resetActiveStoreForTests();
  clearCachedAgentServerInfo();
  delete (window as unknown as Record<string, unknown>)
    .__AGENT_CANVAS_AUTH_REQUIRED__;
});

describe("loadAgentServerInfo", () => {
  it("exposes stable compatibility error contracts through the public alias", () => {
    expect(AGENT_SERVER_UNSUPPORTED_VERSION_ERROR_CODE).toBe(
      "AGENT_SERVER_UNSUPPORTED_VERSION",
    );
    expect(AGENT_SERVER_UNKNOWN_VERSION_ERROR_CODE).toBe(
      "AGENT_SERVER_UNKNOWN_VERSION",
    );
    expect(isAgentServerUnavailableError(new AgentServerUnavailableError())).toBe(
      true,
    );
    expect(
      isAgentServerUnsupportedVersionError(
        new AgentServerUnsupportedVersionError("1.27.0"),
      ),
    ).toBe(true);
    expect(
      isAgentServerUnknownVersionError(
        new AgentServerUnknownVersionError("dev-build"),
      ),
    ).toBe(true);

    (
      window as unknown as Record<string, unknown>
    ).__AGENT_CANVAS_AUTH_REQUIRED__ = true;
    expect(isAgentServerAuthError(httpError(401))).toBe(true);
  });

  it("returns server info when the local backend reports the minimum compatible version", async () => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });

    const result = await loadAgentServerInfo();

    expect(result).toMatchObject({
      version: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
    });
    expect(ServerClient).toHaveBeenCalled();
    expect(SettingsClient).not.toHaveBeenCalled();
  });

  it("omits an empty local API key from the client options", async () => {
    const backendWithoutKey = { ...localBackend, apiKey: "" };
    setRegisteredBackends([backendWithoutKey]);
    setActiveSelection({ backendId: backendWithoutKey.id });

    await loadAgentServerInfo();

    expect(ServerClient).toHaveBeenCalledWith(
      expect.objectContaining({
        host: backendWithoutKey.host,
        timeout: 5000,
      }),
    );
    expect(vi.mocked(ServerClient).mock.calls[0]?.[0]).not.toHaveProperty(
      "apiKey",
    );
  });

  it("throws AgentServerUnsupportedVersionError when the local backend is too old", async () => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    getServerInfoMock.mockResolvedValue({ version: "1.27.1" });

    await expect(loadAgentServerInfo()).rejects.toMatchObject({
      name: AgentServerUnsupportedVersionError.name,
      actualVersion: "1.27.1",
      requiredVersion: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
    });
  });

  it("throws AgentServerUnknownVersionError when the local backend omits its version", async () => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    getServerInfoMock.mockResolvedValue({});

    await expect(loadAgentServerInfo()).rejects.toMatchObject({
      name: AgentServerUnknownVersionError.name,
      actualVersion: null,
      requiredVersion: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
    });
  });

  it("does not borrow a registered local backend when the active backend is cloud", async () => {
    setRegisteredBackends([localBackend, cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    const result = await loadAgentServerInfo();

    expect(result).toBeNull();
    expect(ServerClient).not.toHaveBeenCalled();
  });

  it("throws AgentServerUnavailableError when the registry is empty", async () => {
    // Empty registry — no backends at all (frontend-only with no config).
    setRegisteredBackends([]);

    await expect(loadAgentServerInfo()).rejects.toMatchObject({
      name: AgentServerUnavailableError.name,
      message:
        "No agent server backend is configured yet. Add a backend to get started.",
      details: "No backend configured",
      noBackendConfigured: true,
    });
    expect(ServerClient).not.toHaveBeenCalled();
  });

  it("preserves a 401 returned by the server-info probe", async () => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    const unauthorized = httpError(401);
    getServerInfoMock.mockRejectedValue(unauthorized);

    await expect(loadAgentServerInfo()).rejects.toBe(unauthorized);
  });

  it("wraps an HTTP server-info failure as an unavailable error", async () => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    getServerInfoMock.mockRejectedValue(httpError(503));

    await expect(loadAgentServerInfo()).rejects.toMatchObject({
      name: AgentServerUnavailableError.name,
      details: "HTTP 503",
      noBackendConfigured: false,
    });
  });

  it("wraps a non-Error server-info failure without fabricated details", async () => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    getServerInfoMock.mockRejectedValue("connection closed");

    await expect(loadAgentServerInfo()).rejects.toMatchObject({
      name: AgentServerUnavailableError.name,
      details: null,
    });
  });

  it("validates the key against settings when runtime auth is required", async () => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    (
      window as unknown as Record<string, unknown>
    ).__AGENT_CANVAS_AUTH_REQUIRED__ = true;

    await expect(loadAgentServerInfo()).resolves.toMatchObject({
      version: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
    });

    expect(ServerClient).toHaveBeenCalledWith(
      expect.objectContaining({
        host: localBackend.host,
        apiKey: localBackend.apiKey,
        timeout: 5000,
      }),
    );
    expect(SettingsClient).toHaveBeenCalledWith(
      expect.objectContaining({
        host: localBackend.host,
        apiKey: localBackend.apiKey,
        timeout: 5000,
      }),
    );
    expect(getSettingsMock).toHaveBeenCalledTimes(1);
  });

  it("preserves a 401 returned by the authenticated settings probe", async () => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    (
      window as unknown as Record<string, unknown>
    ).__AGENT_CANVAS_AUTH_REQUIRED__ = true;
    const unauthorized = httpError(401);
    getSettingsMock.mockRejectedValue(unauthorized);

    await expect(loadAgentServerInfo()).rejects.toBe(unauthorized);
  });

  it("continues after a non-401 settings probe failure", async () => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    (
      window as unknown as Record<string, unknown>
    ).__AGENT_CANVAS_AUTH_REQUIRED__ = true;
    const warning = vi.spyOn(console, "warn").mockImplementation(() => {});
    const forbidden = httpError(403);
    getSettingsMock.mockRejectedValue(forbidden);

    await expect(loadAgentServerInfo()).resolves.toMatchObject({
      version: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
    });
    expect(warning).toHaveBeenCalledWith(
      "[agent-server] getSettings() probe failed (non-401):",
      forbidden,
    );
    warning.mockRestore();
  });

  it("uses advertised tools after a successful probe", async () => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    getServerInfoMock.mockResolvedValue({
      version: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
      usable_tools: ["terminal"],
    });

    await loadAgentServerInfo();

    expect(isAgentServerToolAvailable("terminal")).toBe(true);
    expect(isAgentServerToolAvailable("browser_tool_set")).toBe(false);
    clearCachedAgentServerInfo();
    expect(isAgentServerToolAvailable("browser_tool_set")).toBe(true);
  });

  it("clears advertised tools when a later server-info probe fails", async () => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    getServerInfoMock.mockResolvedValue({
      version: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
      usable_tools: ["terminal"],
    });
    await loadAgentServerInfo();
    expect(isAgentServerToolAvailable("browser_tool_set")).toBe(false);

    getServerInfoMock.mockRejectedValue(new Error("connection closed"));

    await expect(loadAgentServerInfo()).rejects.toBeInstanceOf(
      AgentServerUnavailableError,
    );
    expect(isAgentServerToolAvailable("browser_tool_set")).toBe(true);
  });

  it("allows tools when the server does not advertise a tool list", async () => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    getServerInfoMock.mockResolvedValue({
      version: MINIMUM_COMPATIBLE_AGENT_SERVER_VERSION,
      usable_tools: null,
    });

    await loadAgentServerInfo();

    expect(isAgentServerToolAvailable("browser_tool_set")).toBe(true);
    clearCachedAgentServerInfo();
    expect(isAgentServerToolAvailable("browser_tool_set")).toBe(true);
  });
});
