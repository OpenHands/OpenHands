import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { Link, MemoryRouter, useLocation } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { notifyConversationContextChangeRequested } from "#/services/conversation-context-events";
import type {
  CanvasExtensionHost,
  InstalledCanvasExtensionInfo,
} from "#/types/canvas-extension";
import {
  CanvasExtensionsRuntimeProvider,
  useCanvasExtensionsRuntime,
} from "./canvas-extensions-runtime";
import { CanvasExtensionCompanionDock } from "./canvas-extension-companion-surface";

const backend: Backend = {
  id: "extension-backend",
  name: "Extension backend",
  host: "http://127.0.0.1:8000",
  apiKey: "test-key",
  kind: "local",
};

const extension: InstalledCanvasExtensionInfo = {
  name: "demo-extension",
  version: "0.1.0",
  enabled: true,
  source: "github:example/demo",
  resolved_ref: "abc123",
  installed_at: "2026-08-01T00:00:00Z",
  install_path: "/tmp/demo-extension",
  manifest: {
    schema_version: 1,
    name: "demo-extension",
    display_name: "Demo extension",
    version: "0.1.0",
    entrypoint: "dist/extension.js",
    contributes: {
      pages: [
        {
          id: "dashboard",
          title: "Dashboard",
          path: "/dashboard",
          nav_label: "Demo dashboard",
        },
      ],
    },
  },
};

function RuntimeProbe() {
  const runtime = useCanvasExtensionsRuntime();
  const location = useLocation();
  const openChatLabel = "Open chat";
  return (
    <div>
      <span data-testid="page-count">{runtime.pages.length}</span>
      <span data-testid="page-href">{runtime.pages[0]?.href}</span>
      <span data-testid="page-icon">{runtime.pages[0]?.icon}</span>
      <span data-testid="location">{location.pathname}</span>
      <Link to="/conversations/first-cat">{openChatLabel}</Link>
      <span data-testid="runtime-error">
        {runtime.errors.get(extension.name)}
      </span>
    </div>
  );
}

function renderRuntime(
  moduleLoader: (source: string) => Promise<{
    activate: (host: CanvasExtensionHost) => void | (() => void);
  }>,
) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <MemoryRouter>
          <CanvasExtensionsRuntimeProvider moduleLoader={moduleLoader}>
            <div data-testid="runtime-shell">
              <div id="root-outlet">
                <RuntimeProbe />
              </div>
              <CanvasExtensionCompanionDock />
            </div>
          </CanvasExtensionsRuntimeProvider>
        </MemoryRouter>
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

describe("CanvasExtensionsRuntimeProvider", () => {
  beforeEach(() => {
    setRegisteredBackends([backend]);
    setActiveSelection({ backendId: backend.id });
    vi.spyOn(CanvasExtensionsService, "listInstalled").mockResolvedValue([
      extension,
    ]);
    vi.spyOn(CanvasExtensionsService, "fetchBundle").mockResolvedValue(
      "fixture source",
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
    setActiveSelection(null);
    setRegisteredBackends([]);
  });

  it("activates enabled extensions and admits declared page registrations", async () => {
    const disposeActivation = vi.fn();
    const moduleLoader = vi.fn().mockResolvedValue({
      activate: (host: CanvasExtensionHost) => {
        host.registerPage("dashboard", () => undefined);
        return disposeActivation;
      },
    });

    const rendered = renderRuntime(moduleLoader);

    await waitFor(() =>
      expect(screen.getByTestId("page-count")).toHaveTextContent("1"),
    );
    expect(screen.getByTestId("page-href")).toHaveTextContent(
      "/extensions/demo-extension/dashboard",
    );
    expect(CanvasExtensionsService.fetchBundle).toHaveBeenCalledWith(
      extension.name,
      expect.objectContaining({ id: backend.id }),
    );

    rendered.unmount();
    expect(disposeActivation).toHaveBeenCalledTimes(1);
  });

  it("rejects registrations that were not declared in the manifest", async () => {
    const moduleLoader = vi.fn().mockResolvedValue({
      activate: (host: CanvasExtensionHost) => {
        host.registerPage("surprise", () => undefined);
      },
    });

    renderRuntime(moduleLoader);

    await waitFor(() =>
      expect(screen.getByTestId("runtime-error")).toHaveTextContent(
        'registered undeclared page "surprise"',
      ),
    );
    expect(screen.getByTestId("page-count")).toHaveTextContent("0");
  });

  it("degrades gracefully when the backend response has no manifest", async () => {
    vi.mocked(CanvasExtensionsService.listInstalled).mockResolvedValue([
      { ...extension, manifest: null },
    ]);
    const moduleLoader = vi.fn().mockResolvedValue({
      activate: (host: CanvasExtensionHost) => {
        host.registerPage("dashboard", () => undefined);
      },
    });

    renderRuntime(moduleLoader);

    await waitFor(() =>
      expect(screen.getByTestId("runtime-error")).toHaveTextContent(
        'registered undeclared page "dashboard"',
      ),
    );
    expect(screen.getByTestId("page-count")).toHaveTextContent("0");
  });

  it("keeps a companion and activation alive across page navigation", async () => {
    const cleanup = vi.fn();
    const mount = vi.fn(({ container }: { container: HTMLElement }) => {
      container.append(document.createTextNode("Voice connected to first Cat"));
      return cleanup;
    });
    const activate = vi.fn((host: CanvasExtensionHost) => {
      host.registerPage("dashboard", () => undefined, { icon: "cat" });
      host.registerCompanion?.({ id: "voice", mount });
    });
    const rendered = renderRuntime(vi.fn().mockResolvedValue({ activate }));
    const voiceControls = await screen.findByText(
      "Voice connected to first Cat",
    );
    expect(screen.getByTestId("runtime-shell")).toContainElement(voiceControls);
    expect(document.getElementById("root-outlet")).not.toContainElement(
      voiceControls,
    );
    expect(
      screen.getByTestId("canvas-extension-companion-dock"),
    ).toContainElement(voiceControls);
    expect(screen.getByTestId("page-icon")).toHaveTextContent("cat");
    fireEvent.click(screen.getByText("Open chat"));
    await waitFor(() =>
      expect(screen.getByTestId("location")).toHaveTextContent(
        "/conversations/first-cat",
      ),
    );
    expect(mount).toHaveBeenCalledTimes(1);
    expect(activate).toHaveBeenCalledTimes(1);
    expect(cleanup).not.toHaveBeenCalled();
    rendered.unmount();
    expect(cleanup).toHaveBeenCalledTimes(1);
  });

  it("unregisters companions and cleans an asynchronous mount that settles late", async () => {
    let resolveMount!: (cleanup: () => void) => void;
    let unregister!: () => void;
    const cleanup = vi.fn();
    const mount = vi.fn(
      () =>
        new Promise<() => void>((resolve) => {
          resolveMount = resolve;
        }),
    );
    renderRuntime(
      vi.fn().mockResolvedValue({
        activate: (host: CanvasExtensionHost) => {
          unregister = host.registerCompanion!({ id: "voice", mount });
        },
      }),
    );
    await waitFor(() => expect(mount).toHaveBeenCalledTimes(1));
    act(() => unregister());
    expect(
      document.querySelector("[data-app-companion]"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("canvas-extension-companion-dock"),
    ).not.toBeInTheDocument();
    await act(async () => resolveMount(cleanup));
    expect(cleanup).toHaveBeenCalledTimes(1);
  });

  it("scopes context actions to the owning connection and disposes subscriptions", async () => {
    const listener = vi.fn();
    const activate = vi.fn((host: CanvasExtensionHost) => {
      host.onConversationContextChangeRequested?.(listener);
    });
    const rendered = renderRuntime(vi.fn().mockResolvedValue({ activate }));
    await waitFor(() => expect(activate).toHaveBeenCalledTimes(1));
    const scope = { backendId: backend.id, orgId: null, connectionRevision: 0 };
    const event = { conversationId: "first-cat", reason: "condense" as const };
    notifyConversationContextChangeRequested(
      { ...scope, backendId: "other-backend" },
      event,
    );
    notifyConversationContextChangeRequested(
      { ...scope, orgId: "other-org" },
      event,
    );
    notifyConversationContextChangeRequested(
      { ...scope, connectionRevision: 1 },
      event,
    );
    expect(listener).not.toHaveBeenCalled();
    notifyConversationContextChangeRequested(scope, event);
    expect(listener).toHaveBeenCalledExactlyOnceWith(event);
    rendered.unmount();
    notifyConversationContextChangeRequested(scope, event);
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("tears down on backend replacement and keeps late cleanup bound to its owner", async () => {
    let originalHost!: CanvasExtensionHost;
    const cleanup = vi.fn();
    const request = vi
      .spyOn(CanvasExtensionsService, "requestAgentServer")
      .mockResolvedValue({ success: true });
    renderRuntime(
      vi.fn().mockResolvedValue({
        activate: (host: CanvasExtensionHost) => {
          if (!originalHost) originalHost = host;
          host.registerCompanion?.({
            id: "voice",
            mount: ({ container }) => {
              container.append(document.createTextNode(host.backend.id));
              return cleanup;
            },
          });
        },
      }),
    );
    await screen.findByText(backend.id);
    act(() => {
      setRegisteredBackends([backend, { ...backend, id: "other-backend" }]);
      setActiveSelection({ backendId: "other-backend" });
    });
    await screen.findByText("other-backend");
    expect(cleanup).toHaveBeenCalledTimes(1);
    const release = {
      method: "DELETE" as const,
      path: "/api/conversations/first-cat/voice/realtime/old-call",
    };
    await originalHost.agentServer.request(release);
    expect(request).toHaveBeenCalledWith(
      release,
      expect.objectContaining({ id: backend.id }),
    );
  });
});
