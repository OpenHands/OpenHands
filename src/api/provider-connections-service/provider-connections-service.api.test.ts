import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { AgentServerClient } from "@openhands/typescript-client/clients";
import ProviderConnectionsService from "./provider-connections-service.api";

const getMock = vi.hoisted(() => vi.fn());
const postMock = vi.hoisted(() => vi.fn());
const patchMock = vi.hoisted(() => vi.fn());
const deleteMock = vi.hoisted(() => vi.fn());
const closeMock = vi.hoisted(() => vi.fn());

vi.mock("@openhands/typescript-client/clients", () => ({
  AgentServerClient: vi.fn(function AgentServerClientMock() {
    return {
      get: getMock,
      post: postMock,
      patch: patchMock,
      delete: deleteMock,
      close: closeMock,
    };
  }),
}));

const localBackend: Backend = {
  id: "local-test",
  name: "Local test backend",
  host: "http://localhost:3000",
  apiKey: "test-session-key",
  kind: "local",
};

const connection = {
  id: "conn-1",
  display_name: "My OpenAI",
  provider: "openai",
  base_url: null,
  created_at: 1,
  updated_at: 2,
  api_key_set: true,
};

const PATH = "/api/llm/provider-connections";

describe("ProviderConnectionsService", () => {
  beforeEach(() => {
    getMock.mockReset();
    postMock.mockReset();
    patchMock.mockReset();
    deleteMock.mockReset();
    closeMock.mockReset();
    vi.mocked(AgentServerClient).mockClear();
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
  });

  afterEach(() => {
    setActiveSelection(null);
    setRegisteredBackends([]);
  });

  it("lists connections via the typed client and closes it", async () => {
    getMock.mockResolvedValue([connection]);

    const result = await ProviderConnectionsService.list();

    expect(result).toEqual([connection]);
    expect(vi.mocked(AgentServerClient)).toHaveBeenCalledWith({
      host: "http://localhost:3000",
      apiKey: "test-session-key",
    });
    expect(getMock).toHaveBeenCalledWith(PATH, { responseType: "json" });
    expect(closeMock).toHaveBeenCalled();
  });

  it("creates a connection with the request body", async () => {
    postMock.mockResolvedValue(connection);

    const request = {
      display_name: "My OpenAI",
      provider: "openai",
      api_key: "sk-123",
      base_url: null,
    };
    const result = await ProviderConnectionsService.create(request);

    expect(result).toEqual(connection);
    expect(postMock).toHaveBeenCalledWith(PATH, request, {
      responseType: "json",
    });
    expect(closeMock).toHaveBeenCalled();
  });

  it("updates a connection at the id-scoped path", async () => {
    patchMock.mockResolvedValue(connection);

    const result = await ProviderConnectionsService.update("conn-1", {
      display_name: "Renamed",
    });

    expect(result).toEqual(connection);
    expect(patchMock).toHaveBeenCalledWith(
      `${PATH}/conn-1`,
      { display_name: "Renamed" },
      { responseType: "json" },
    );
    expect(closeMock).toHaveBeenCalled();
  });

  it("url-encodes the id when deleting", async () => {
    deleteMock.mockResolvedValue(connection);

    await ProviderConnectionsService.delete("a b/c");

    expect(deleteMock).toHaveBeenCalledWith(`${PATH}/a%20b%2Fc`, {
      responseType: "json",
    });
    expect(closeMock).toHaveBeenCalled();
  });

  it("closes the client even when the request throws", async () => {
    getMock.mockRejectedValue(new Error("boom"));

    await expect(ProviderConnectionsService.list()).rejects.toThrow("boom");
    expect(closeMock).toHaveBeenCalled();
  });
});
