import { http, HttpResponse } from "msw";
import type {
  CanvasExtensionManifest,
  InstallCanvasExtensionRequest,
  InstalledCanvasExtensionInfo,
} from "#/types/canvas-extension";
import demoManifestSource from "#/fixtures/canvas-extensions/demo-page/canvas-extension.json?raw";
// eslint-disable-next-line import-x/extensions -- Vite requires the real filename before ?raw.
import demoBundle from "#/fixtures/canvas-extensions/demo-page/extension.js?raw";

export const CANVAS_EXTENSION_DEMO_SOURCE =
  "src/fixtures/canvas-extensions/demo-page";

const demoManifest = JSON.parse(demoManifestSource) as CanvasExtensionManifest;

const MOCK_STORAGE_KEY = "openhands-canvas-extension-dev-installation";

function readPersistedExtension(): InstalledCanvasExtensionInfo | null {
  try {
    const value = globalThis.sessionStorage?.getItem(MOCK_STORAGE_KEY);
    return value ? (JSON.parse(value) as InstalledCanvasExtensionInfo) : null;
  } catch {
    return null;
  }
}

let installedExtension = readPersistedExtension();

function persistInstalledExtension(
  extension: InstalledCanvasExtensionInfo | null,
): void {
  installedExtension = extension;
  try {
    if (extension) {
      globalThis.sessionStorage?.setItem(
        MOCK_STORAGE_KEY,
        JSON.stringify(extension),
      );
    } else {
      globalThis.sessionStorage?.removeItem(MOCK_STORAGE_KEY);
    }
  } catch {
    // The module variable still provides an in-memory fallback.
  }
}

function isDemoSource(source: string): boolean {
  return source.replaceAll("\\", "/").endsWith(CANVAS_EXTENSION_DEMO_SOURCE);
}

function getRequestedName(
  value: string | readonly string[] | undefined,
): string {
  if (!value) return "";
  return decodeURIComponent(Array.isArray(value) ? value[0] : value);
}

function extensionNotFound(name: string) {
  return HttpResponse.json(
    { detail: `Canvas extension '${name}' is not installed.` },
    { status: 404 },
  );
}

function getInstalledExtension(
  name: string,
): InstalledCanvasExtensionInfo | null {
  if (installedExtension?.name !== name) return null;
  return installedExtension;
}

export function resetCanvasExtensionsMockData(): void {
  persistInstalledExtension(null);
}

/**
 * Browser-development implementation of the Canvas Extension API contract.
 * It intentionally supports only the checked-in demo fixture. Production and
 * ordinary dev stacks continue to use the active Agent Server exclusively.
 */
export const CANVAS_EXTENSIONS_HANDLERS = [
  http.get("*/api/canvas-extensions/installed", () =>
    HttpResponse.json({
      canvas_extensions: installedExtension ? [installedExtension] : [],
    }),
  ),

  http.post("*/api/canvas-extensions/install", async ({ request }) => {
    const body = (await request.json()) as InstallCanvasExtensionRequest;
    if (!body.source || !isDemoSource(body.source)) {
      return HttpResponse.json(
        {
          detail: `Mock mode only installs '${CANVAS_EXTENSION_DEMO_SOURCE}'.`,
        },
        { status: 400 },
      );
    }

    if (installedExtension && !body.force) {
      return HttpResponse.json(
        {
          detail: `Canvas extension '${demoManifest.name}' is already installed.`,
        },
        { status: 409 },
      );
    }

    persistInstalledExtension({
      name: demoManifest.name,
      version: demoManifest.version,
      description: demoManifest.description,
      enabled: false,
      source: body.source,
      resolved_ref: body.ref || "mock-working-tree",
      repo_path: body.repo_path || null,
      installed_at: new Date().toISOString(),
      install_path: `/mock/canvas-extensions/${demoManifest.name}`,
      // The real backend only returns `manifest` once OSS-10048 lands; the
      // mock previews that shape so the demo fixture's pages work in dev mode.
      manifest: demoManifest,
    });

    return HttpResponse.json(installedExtension);
  }),

  http.get("*/api/canvas-extensions/installed/:name/bundle", ({ params }) => {
    const name = getRequestedName(params.name);
    if (!getInstalledExtension(name)) return extensionNotFound(name);
    return new HttpResponse(demoBundle, {
      headers: {
        "Content-Type": "application/javascript; charset=utf-8",
        "Cache-Control": "no-cache",
      },
    });
  }),

  http.get("*/api/canvas-extensions/installed/:name", ({ params }) => {
    const name = getRequestedName(params.name);
    const extension = getInstalledExtension(name);
    return extension ? HttpResponse.json(extension) : extensionNotFound(name);
  }),

  http.patch(
    "*/api/canvas-extensions/installed/:name",
    async ({ params, request }) => {
      const name = getRequestedName(params.name);
      const extension = getInstalledExtension(name);
      if (!extension) return extensionNotFound(name);

      const body = (await request.json()) as { enabled?: unknown };
      if (typeof body.enabled !== "boolean") {
        return HttpResponse.json(
          { detail: "'enabled' must be a boolean." },
          { status: 422 },
        );
      }

      persistInstalledExtension({ ...extension, enabled: body.enabled });
      return HttpResponse.json({ name, enabled: body.enabled });
    },
  ),

  http.delete("*/api/canvas-extensions/installed/:name", ({ params }) => {
    const name = getRequestedName(params.name);
    if (!getInstalledExtension(name)) return extensionNotFound(name);
    persistInstalledExtension(null);
    return HttpResponse.json({ message: "Canvas extension uninstalled." });
  }),
];
