import { AgentServerClient } from "@openhands/typescript-client/clients";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import CanvasExtensionsService, {
  CanvasExtensionsUnsupportedError,
} from "#/api/canvas-extensions-service";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import type { InstalledCanvasExtensionInfo } from "#/types/canvas-extension";

vi.mock("@openhands/typescript-client/clients", () => ({
  AgentServerClient: vi.fn(),
}));

const get = vi.fn();
const post = vi.fn();
const patch = vi.fn();
const remove = vi.fn();
const request = vi.fn();

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://127.0.0.1:8000",
  apiKey: "session-key",
  kind: "local",
};

const extension: InstalledCanvasExtensionInfo = {
  name: "demo-extension",
  version: "0.1.0",
  description: "Demo",
  enabled: false,
  source: "github:example/extensions",
  resolved_ref: "abc123",
  repo_path: "extensions/demo",
  installed_at: "2026-08-01T00:00:00Z",
  install_path: "/tmp/extensions/demo-extension",
  manifest: {
    schema_version: 1,
    name: "demo-extension",
    version: "0.1.0",
    entrypoint: "dist/extension.js",
    contributes: {
      pages: [{ id: "home", title: "Home", path: "home" }],
    },
  },
};

beforeEach(() => {
  vi.clearAllMocks();
  __resetActiveStoreForTests();
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
  vi.mocked(AgentServerClient).mockImplementation(
    function MockAgentServerClient() {
      return {
        get,
        post,
        patch,
        delete: remove,
        request,
        close: vi.fn(),
      } as unknown as AgentServerClient;
    } as unknown as typeof AgentServerClient,
  );
});

afterEach(() => {
  setActiveSelection(null);
  setRegisteredBackends([]);
  __resetActiveStoreForTests();
});

describe("CanvasExtensionsService", () => {
  it("lists extensions installed on the active Agent Server", async () => {
    get.mockResolvedValue({ canvas_extensions: [extension] });

    await expect(CanvasExtensionsService.listInstalled()).resolves.toEqual([
      extension,
    ]);
    expect(get).toHaveBeenCalledWith("/api/canvas-extensions/installed");
  });

  it("preserves install coordinates without allowing an enabled flag", async () => {
    post.mockResolvedValue(extension);

    await CanvasExtensionsService.install({
      source: "github:example/extensions",
      ref: "main",
      repo_path: "extensions/demo",
    });

    expect(post).toHaveBeenCalledWith("/api/canvas-extensions/install", {
      source: "github:example/extensions",
      ref: "main",
      repo_path: "extensions/demo",
    });
  });

  it("maps a missing router to an explicit unsupported-backend error", async () => {
    get.mockRejectedValue(
      Object.assign(new Error("Not Found"), {
        name: "HttpError",
        status: 404,
      }),
    );

    await expect(CanvasExtensionsService.listInstalled()).rejects.toEqual(
      expect.objectContaining<Partial<CanvasExtensionsUnsupportedError>>({
        name: "CanvasExtensionsUnsupportedError",
        reason: "missing-api",
      }),
    );
  });

  it("does not call the Agent Server protocol for a Cloud backend", async () => {
    const cloud: Backend = {
      ...localBackend,
      id: "cloud",
      name: "Cloud",
      kind: "cloud",
    };
    setRegisteredBackends([cloud]);
    setActiveSelection({ backendId: cloud.id });

    await expect(CanvasExtensionsService.listInstalled()).rejects.toEqual(
      expect.objectContaining({ reason: "cloud-backend" }),
    );
    expect(AgentServerClient).not.toHaveBeenCalled();
  });

  it("fetches the bundle as authenticated text from the captured backend", async () => {
    get.mockResolvedValue("export const activate = () => {};");

    await expect(
      CanvasExtensionsService.fetchBundle(extension.name, localBackend),
    ).resolves.toContain("activate");

    expect(AgentServerClient).toHaveBeenCalledWith({
      host: localBackend.host,
      apiKey: localBackend.apiKey,
      timeout: 60000,
    });
    expect(get).toHaveBeenCalledWith(
      "/api/canvas-extensions/installed/demo-extension/bundle",
      { responseType: "text" },
    );
  });

  it("encodes extension names in mutation paths", async () => {
    patch.mockResolvedValue({ name: "a/b", enabled: true });

    await CanvasExtensionsService.setEnabled("a/b", true);

    expect(patch).toHaveBeenCalledWith(
      "/api/canvas-extensions/installed/a%2Fb",
      { enabled: true },
    );
  });

  it("does not send the backend session key to an absolute request URL", async () => {
    await expect(
      CanvasExtensionsService.requestAgentServer({
        path: "https://example.com/collect",
      }),
    ).rejects.toThrow("root-relative path");
    expect(AgentServerClient).not.toHaveBeenCalled();
  });
});
