import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  FileClient,
  PluginsClient,
} from "@openhands/typescript-client/clients";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "./backend-registry/active-store";
import PluginsService from "./plugins-service";

vi.mock("@openhands/typescript-client/clients", () => ({
  PluginsClient: vi.fn(),
  FileClient: vi.fn(),
}));

const getPluginsMarketplace = vi.fn();
const getPlugins = vi.fn();
const downloadFile = vi.fn();
const close = vi.fn();

function useBackend(kind: "local" | "cloud"): void {
  setRegisteredBackends([
    {
      id: kind,
      name: kind,
      host: "http://127.0.0.1:8001",
      apiKey: "session-key",
      kind,
    },
  ]);
  setActiveSelection({ backendId: kind, orgId: null });
}

describe("PluginsService.getPluginsMarketplace", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(PluginsClient).mockImplementation(function MockPluginsClient() {
      return { getPluginsMarketplace, close } as unknown as PluginsClient;
    } as unknown as typeof PluginsClient);
  });

  it("returns the catalog from the local agent-server", async () => {
    useBackend("local");
    const plugin = {
      name: "city-weather",
      description: "Weather plugin",
      source: "github:OpenHands/extensions",
      ref: null,
      repo_path: "plugins/city-weather",
      installed: false,
    };
    getPluginsMarketplace.mockResolvedValue({ plugins: [plugin] });

    const result = await PluginsService.getPluginsMarketplace();

    expect(result).toEqual([plugin]);
    expect(getPluginsMarketplace).toHaveBeenCalledTimes(1);
  });

  it("returns an empty catalog on a cloud backend without calling the client", async () => {
    useBackend("cloud");

    const result = await PluginsService.getPluginsMarketplace();

    expect(result).toEqual([]);
    expect(PluginsClient).not.toHaveBeenCalled();
  });

  it("returns an empty catalog when the local request fails", async () => {
    useBackend("local");
    getPluginsMarketplace.mockRejectedValue(new Error("unreachable"));

    const result = await PluginsService.getPluginsMarketplace();

    expect(result).toEqual([]);
  });
});

describe("PluginsService.getLocalPlugins", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(PluginsClient).mockImplementation(function MockPluginsClient() {
      return { getPlugins, close } as unknown as PluginsClient;
    } as unknown as typeof PluginsClient);
  });

  it("requests user-level local plugins from the local agent-server", async () => {
    useBackend("local");
    const plugin = {
      name: "hello-local",
      version: "1.0.0",
      description: "A local plugin",
    };
    getPlugins.mockResolvedValue({ plugins: [plugin] });

    const result = await PluginsService.getLocalPlugins();

    expect(result).toEqual([plugin]);
    expect(getPlugins).toHaveBeenCalledWith({
      load_user: true,
      load_project: false,
    });
  });

  it("returns an empty list on a cloud backend without calling the client", async () => {
    useBackend("cloud");

    const result = await PluginsService.getLocalPlugins();

    expect(result).toEqual([]);
    expect(PluginsClient).not.toHaveBeenCalled();
  });
});

describe("PluginsService.getPluginFileContent", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(FileClient).mockImplementation(function MockFileClient() {
      return { downloadFile, close } as unknown as FileClient;
    } as unknown as typeof FileClient);
  });

  it("downloads the file under the plugin directory and decodes it as text", async () => {
    useBackend("local");
    downloadFile.mockResolvedValue(new TextEncoder().encode("# Hello").buffer);

    const result = await PluginsService.getPluginFileContent(
      "/plugins/demo",
      "docs/README.md",
    );

    expect(result).toEqual({ kind: "text", text: "# Hello" });
    expect(downloadFile).toHaveBeenCalledWith("/plugins/demo/docs/README.md");
  });

  it("flags content containing NUL bytes as binary", async () => {
    useBackend("local");
    downloadFile.mockResolvedValue(
      new Uint8Array([0x89, 0x50, 0x00, 0x47]).buffer,
    );

    const result = await PluginsService.getPluginFileContent(
      "/plugins/demo",
      "logo.png",
    );

    expect(result).toEqual({ kind: "binary", text: null });
  });

  it("rejects on a cloud backend without calling the client", async () => {
    useBackend("cloud");

    await expect(
      PluginsService.getPluginFileContent("/plugins/demo", "README.md"),
    ).rejects.toThrow();
    expect(FileClient).not.toHaveBeenCalled();
  });
});
