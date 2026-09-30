import { AgentServerClient } from "@openhands/typescript-client/clients";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { callCloudProxy } from "#/api/cloud/proxy";
import ToolCatalogService, {
  type ToolCatalogEntry,
} from "#/api/tool-catalog-service/tool-catalog-service.api";

vi.mock("@openhands/typescript-client/clients", () => ({
  AgentServerClient: vi.fn(),
}));

vi.mock("#/api/cloud/proxy", () => ({
  callCloudProxy: vi.fn(),
}));

const get = vi.fn();

const localBackend: Backend = {
  id: "local",
  name: "local",
  host: "http://127.0.0.1:8000",
  apiKey: "session-key",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "cloud",
  name: "cloud",
  host: "https://app.example.test",
  apiKey: "cloud-key",
  kind: "cloud",
};

const catalogEntry = (
  overrides: Partial<ToolCatalogEntry> = {},
): ToolCatalogEntry => ({
  name: "terminal",
  user_selectable: true,
  usable: true,
  in_default_set: true,
  description: "Run shell commands.",
  ...overrides,
});

beforeEach(() => {
  vi.clearAllMocks();
  setRegisteredBackends([localBackend, cloudBackend]);
  setActiveSelection({ backendId: localBackend.id });
  vi.mocked(AgentServerClient).mockImplementation(
    function MockAgentServerClient() {
      return {
        get,
      } as unknown as AgentServerClient;
    } as unknown as typeof AgentServerClient,
  );
});

afterEach(() => {
  setActiveSelection(null);
  setRegisteredBackends([]);
  __resetActiveStoreForTests();
});

describe("ToolCatalogService.getCatalog", () => {
  it("returns the server's catalog unchanged when every entry is well-formed", async () => {
    const tools = [
      catalogEntry(),
      catalogEntry({ name: "glob", in_default_set: false }),
    ];
    get.mockResolvedValue({ tools });

    await expect(ToolCatalogService.getCatalog()).resolves.toEqual(tools);
    expect(get).toHaveBeenCalledWith("/api/tools/catalog");
  });

  it("fails loudly on a catalog entry that omits a discriminator field", async () => {
    get.mockResolvedValue({
      tools: [
        { name: "terminal", user_selectable: true, usable: true },
        catalogEntry(),
      ],
    });

    await expect(ToolCatalogService.getCatalog()).rejects.toThrow(
      'malformed tool catalog entry (name="terminal")',
    );
    await expect(ToolCatalogService.getCatalog()).rejects.toThrow(
      "refusing to pick tools from it",
    );
  });

  it("fails loudly when the response omits the tools array entirely", async () => {
    get.mockResolvedValue({});

    await expect(ToolCatalogService.getCatalog()).rejects.toThrow(
      "malformed tool catalog (missing the tools array)",
    );
  });

  it("routes cloud backends through the cloud proxy and validates its payload too", async () => {
    setActiveSelection({ backendId: cloudBackend.id });
    vi.mocked(callCloudProxy).mockResolvedValue({
      tools: [{ name: "terminal" }],
    });

    await expect(ToolCatalogService.getCatalog()).rejects.toThrow(
      "malformed tool catalog entry",
    );
    expect(callCloudProxy).toHaveBeenCalledWith({
      backend: cloudBackend,
      method: "GET",
      path: "/api/tools/catalog",
    });
  });
});
