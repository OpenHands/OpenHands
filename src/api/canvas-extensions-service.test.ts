import { AgentServerClient } from "@openhands/typescript-client/clients";
import * as AgentServerClients from "@openhands/typescript-client/clients";
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
  CanvasExtensionsClient: vi.fn(),
}));

const get = vi.fn();
const post = vi.fn();
const patch = vi.fn();
const remove = vi.fn();
const request = vi.fn();
const getServerInfo = vi.fn();
const createAppBackendSession = vi.fn();
const revokeAppBackendSession = vi.fn();
const closeAppBackendClient = vi.fn();

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
        server: { getServerInfo },
        close: vi.fn(),
      } as unknown as AgentServerClient;
    } as unknown as typeof AgentServerClient,
  );
  const CanvasExtensionsClientMock = (
    AgentServerClients as typeof AgentServerClients & {
      CanvasExtensionsClient: ReturnType<typeof vi.fn>;
    }
  ).CanvasExtensionsClient;
  CanvasExtensionsClientMock.mockImplementation(
    function MockCanvasExtensionsClient() {
      return {
        createAppBackendSession,
        revokeAppBackendSession,
        close: closeAppBackendClient,
      };
    } as never,
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

  it.each([
    {
      source: "/repository-name/",
      repoPath: "/demo-extension",
      expectedSource: "/repository-name/demo-extension",
    },
    {
      source: "/repository-name",
      repoPath: "demo-extension",
      expectedSource: "/repository-name/demo-extension",
    },
    {
      source: "/repository-name/demo-extension",
      repoPath: null,
      expectedSource: "/repository-name/demo-extension",
    },
  ])(
    "resolves a local extension from source $source and path $repoPath",
    async ({ source, repoPath, expectedSource }) => {
      post.mockResolvedValue(extension);

      await CanvasExtensionsService.install({
        source,
        ref: null,
        repo_path: repoPath,
      });

      expect(post).toHaveBeenCalledWith("/api/canvas-extensions/install", {
        source: expectedSource,
        ref: null,
        repo_path: null,
      });
    },
  );

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

  it("returns no app view helper when the captured backend lacks bridge capability", async () => {
    getServerInfo.mockResolvedValue({ capabilities: [] });

    await expect(
      CanvasExtensionsService.createAppBackendViewClient(
        extension.name,
        localBackend,
      ),
    ).resolves.toBeNull();
    expect(createAppBackendSession).not.toHaveBeenCalled();
  });

  it("creates and revokes app sessions through the discovered bridge ingress", async () => {
    getServerInfo.mockResolvedValue({
      capabilities: ["canvas_app_backend_bridge_v1"],
      app_backend_ingress_url: "https://apps.example.test",
    });
    createAppBackendSession.mockResolvedValue({
      ingress_url: "https://apps.example.test/app-backends/demo-extension/",
      expires_at: "2026-09-23T16:00:00Z",
      iframe_sandbox:
        "allow-forms allow-modals allow-popups allow-same-origin allow-scripts",
    });

    const viewClient = await CanvasExtensionsService.createAppBackendViewClient(
      extension.name,
      localBackend,
    );

    expect(viewClient).not.toBeNull();
    await expect(
      viewClient!.createSession({ folder: "/workspace/my project" }),
    ).resolves.toEqual({
      url: "https://apps.example.test/app-backends/demo-extension/?folder=%2Fworkspace%2Fmy+project",
      expiresAt: "2026-09-23T16:00:00Z",
      iframeSandbox:
        "allow-forms allow-modals allow-popups allow-same-origin allow-scripts",
    });
    await viewClient!.revokeSession();
    viewClient!.dispose();
    const CanvasExtensionsClient = (
      AgentServerClients as typeof AgentServerClients & {
        CanvasExtensionsClient: ReturnType<typeof vi.fn>;
      }
    ).CanvasExtensionsClient;
    expect(CanvasExtensionsClient).toHaveBeenCalledWith({
      host: localBackend.host,
      apiKey: localBackend.apiKey,
      timeout: 60000,
      workingDir: expect.any(String),
      appBackendIngressUrl: "https://apps.example.test/",
    });
    expect(createAppBackendSession).toHaveBeenCalledWith(extension.name);
    expect(revokeAppBackendSession).toHaveBeenCalledWith(extension.name);
    expect(closeAppBackendClient).toHaveBeenCalledTimes(1);
  });

  it("rejects credential-like App view query keys", async () => {
    getServerInfo.mockResolvedValue({
      capabilities: ["canvas_app_backend_bridge_v1"],
      app_backend_ingress_url: "https://apps.example.test",
    });
    createAppBackendSession.mockResolvedValue({
      ingress_url: "https://apps.example.test/app-backends/demo-extension/",
      expires_at: "2026-09-23T16:00:00Z",
      iframe_sandbox: "allow-scripts",
    });
    const viewClient = await CanvasExtensionsService.createAppBackendViewClient(
      extension.name,
      localBackend,
    );

    await expect(
      viewClient!.createSession({ token: "secret" }),
    ).rejects.toThrow("Unsupported or sensitive");
    expect(revokeAppBackendSession).toHaveBeenCalledWith(extension.name);
  });

  it("rejects and revokes a session outside the discovered ingress origin", async () => {
    getServerInfo.mockResolvedValue({
      capabilities: ["canvas_app_backend_bridge_v1"],
      app_backend_ingress_url: "https://apps.example.test",
    });
    createAppBackendSession.mockResolvedValue({
      ingress_url: "https://attacker.example/app-backends/demo-extension/",
      expires_at: "2026-09-23T16:00:00Z",
      iframe_sandbox: "allow-scripts",
    });
    const viewClient = await CanvasExtensionsService.createAppBackendViewClient(
      extension.name,
      localBackend,
    );

    await expect(viewClient!.createSession()).rejects.toThrow(
      "session URL is invalid",
    );
    expect(revokeAppBackendSession).toHaveBeenCalledWith(extension.name);
  });
});
