import { AgentServerClient } from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { isSdkHttpStatusError } from "#/api/agent-server-compatibility";
import {
  getActiveBackend,
  isNoBackend,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import type {
  CanvasExtensionAgentServerRequest,
  InstallCanvasExtensionRequest,
  InstalledCanvasExtensionInfo,
} from "#/types/canvas-extension";
import { isValidCanvasExtensionIconPath } from "#/utils/canvas-extension-icon";

export const CANVAS_EXTENSIONS_BASE_PATH = "/api/canvas-extensions";

export type CanvasExtensionsUnsupportedReason =
  | "no-backend"
  | "cloud-backend"
  | "missing-api";

export class CanvasExtensionsUnsupportedError extends Error {
  readonly reason: CanvasExtensionsUnsupportedReason;

  constructor(reason: CanvasExtensionsUnsupportedReason) {
    const message =
      reason === "no-backend"
        ? "Add an Agent Server backend to use Canvas Extensions."
        : reason === "cloud-backend"
          ? "Canvas Extensions are not available on Cloud backends yet."
          : "This Agent Server does not support Canvas Extensions yet. Upgrade the backend and try again.";
    super(message);
    this.name = "CanvasExtensionsUnsupportedError";
    this.reason = reason;
  }
}

export function isCanvasExtensionsUnsupportedError(
  error: unknown,
): error is CanvasExtensionsUnsupportedError {
  return (
    error instanceof CanvasExtensionsUnsupportedError ||
    (typeof error === "object" &&
      error !== null &&
      "name" in error &&
      error.name === "CanvasExtensionsUnsupportedError")
  );
}

function requireSupportedBackend(): void {
  const { backend } = getActiveBackend();
  if (isNoBackend(backend)) {
    throw new CanvasExtensionsUnsupportedError("no-backend");
  }
  if (backend.kind === "cloud") {
    throw new CanvasExtensionsUnsupportedError("cloud-backend");
  }
}

function getClient(): AgentServerClient {
  requireSupportedBackend();
  return new AgentServerClient(getAgentServerClientOptions());
}

function getClientForBackend(backend: Backend): AgentServerClient {
  if (isNoBackend(backend)) {
    throw new CanvasExtensionsUnsupportedError("no-backend");
  }
  if (backend.kind === "cloud") {
    throw new CanvasExtensionsUnsupportedError("cloud-backend");
  }
  return new AgentServerClient({
    host: backend.host,
    ...(backend.apiKey ? { apiKey: backend.apiKey } : {}),
    timeout: 60000,
  });
}

async function mapUnsupported<T>(operation: () => Promise<T>): Promise<T> {
  try {
    return await operation();
  } catch (error) {
    if (isSdkHttpStatusError(error, 404)) {
      throw new CanvasExtensionsUnsupportedError("missing-api");
    }
    throw error;
  }
}

function installedExtensionPath(name: string): string {
  return `${CANVAS_EXTENSIONS_BASE_PATH}/installed/${encodeURIComponent(name)}`;
}

class CanvasExtensionsService {
  static async listInstalled(): Promise<InstalledCanvasExtensionInfo[]> {
    const client = getClient();
    const response = await mapUnsupported(() =>
      client.get<{ canvas_extensions: InstalledCanvasExtensionInfo[] }>(
        `${CANVAS_EXTENSIONS_BASE_PATH}/installed`,
      ),
    );
    return response.canvas_extensions ?? [];
  }

  static async install(
    request: InstallCanvasExtensionRequest,
  ): Promise<InstalledCanvasExtensionInfo> {
    const client = getClient();
    return mapUnsupported(() =>
      client.post<InstalledCanvasExtensionInfo>(
        `${CANVAS_EXTENSIONS_BASE_PATH}/install`,
        request,
      ),
    );
  }

  static async setEnabled(
    name: string,
    enabled: boolean,
  ): Promise<{ name: string; enabled: boolean }> {
    const client = getClient();
    return mapUnsupported(() =>
      client.patch<{ name: string; enabled: boolean }>(
        installedExtensionPath(name),
        { enabled },
      ),
    );
  }

  static async uninstall(name: string): Promise<{ message: string }> {
    const client = getClient();
    return mapUnsupported(() =>
      client.delete<{ message: string }>(installedExtensionPath(name)),
    );
  }

  /**
   * Build the path that serves a manifest-declared SVG icon from the installed
   * extension's root. Returns `null` for a missing or unsafe path so callers
   * degrade to the default icon instead of issuing a bogus request.
   */
  static buildIconUrl(name: string, iconPath: string): string | null {
    if (!isValidCanvasExtensionIconPath(iconPath)) return null;
    return `${installedExtensionPath(name)}/file?path=${encodeURIComponent(
      iconPath.trim(),
    )}`;
  }

  /**
   * Fetch a manifest-declared SVG icon through the active backend's typed
   * client, so the request carries the session API key and targets the
   * correct backend host (the same sanctioned path as every other extension
   * API call). Returns `null` when the declared path is unsafe or the
   * backend cannot serve the asset, so callers fall back to the default
   * icon instead of breaking extension loading.
   */
  static async fetchIcon(name: string, iconPath: string): Promise<Blob | null> {
    const url = this.buildIconUrl(name, iconPath);
    if (!url) return null;
    try {
      const client = getClient();
      return await client.get<Blob>(url, { responseType: "blob" });
    } catch {
      return null;
    }
  }

  static async fetchBundle(name: string, backend?: Backend): Promise<string> {
    const client = backend ? getClientForBackend(backend) : getClient();
    return mapUnsupported(() =>
      client.get<string>(`${installedExtensionPath(name)}/bundle`, {
        responseType: "text",
      }),
    );
  }

  static async requestAgentServer<T = unknown>(
    request: CanvasExtensionAgentServerRequest,
    backend?: Backend,
  ): Promise<T> {
    if (!request.path.startsWith("/") || request.path.startsWith("//")) {
      throw new Error(
        "Canvas Extension Agent Server requests must use a root-relative path.",
      );
    }
    const client = backend ? getClientForBackend(backend) : getClient();
    return client.request<T>({
      method: request.method ?? "GET",
      path: request.path,
      body: request.body,
      headers: request.headers,
    });
  }
}

export default CanvasExtensionsService;
