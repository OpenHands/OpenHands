import { ToolClient } from "@openhands/typescript-client/clients";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import ToolCatalogService, {
  type ToolCatalogEntry,
} from "#/api/tool-catalog-service/tool-catalog-service.api";

vi.mock("@openhands/typescript-client/clients", () => ({
  ToolClient: vi.fn(),
}));

const getToolCatalog = vi.fn();

const localBackend: Backend = {
  id: "local",
  name: "local",
  host: "http://127.0.0.1:8000",
  apiKey: "session-key",
  kind: "local",
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
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
  vi.mocked(ToolClient).mockImplementation(function MockToolClient() {
    return { getToolCatalog } as unknown as ToolClient;
  } as unknown as typeof ToolClient);
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
    getToolCatalog.mockResolvedValue({ tools });

    await expect(ToolCatalogService.getCatalog()).resolves.toEqual(tools);
    expect(ToolClient).toHaveBeenCalledWith(
      expect.objectContaining({ host: localBackend.host }),
    );
  });

  it("fails loudly on a catalog entry that omits a discriminator field", async () => {
    getToolCatalog.mockResolvedValue({
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
    getToolCatalog.mockResolvedValue({});

    await expect(ToolCatalogService.getCatalog()).rejects.toThrow(
      "malformed tool catalog (missing the tools array)",
    );
  });
});
