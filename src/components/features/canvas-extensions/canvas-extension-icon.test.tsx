import { render, screen, fireEvent } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AgentServerClient } from "@openhands/typescript-client/clients";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import type { InstalledCanvasExtensionInfo } from "#/types/canvas-extension";
import { CanvasExtensionIcon } from "./canvas-extension-icon";

vi.mock("@openhands/typescript-client/clients", () => ({
  AgentServerClient: vi.fn(),
}));

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://127.0.0.1:8000",
  apiKey: "session-key",
  kind: "local",
};

const get = vi.fn();

function extensionWithIcon(
  icon: string | null | undefined,
): InstalledCanvasExtensionInfo {
  return {
    name: "demo-page",
    version: "0.1.0",
    enabled: true,
    source: "github:example/demo",
    installed_at: "2026-08-01T00:00:00Z",
    install_path: "/tmp/demo-page",
    manifest: {
      schema_version: 1,
      name: "demo-page",
      version: "0.1.0",
      ...(icon === undefined ? {} : { icon }),
      entrypoint: "extension.js",
    },
  };
}

function svgIconBlob(): Blob {
  return new Blob(['<svg xmlns="http://www.w3.org/2000/svg"/>'], {
    type: "image/svg+xml",
  });
}

beforeEach(() => {
  get.mockReset();
  vi.mocked(AgentServerClient).mockImplementation(
    function MockAgentServerClient() {
      return { get } as unknown as AgentServerClient;
    } as unknown as typeof AgentServerClient,
  );
  __resetActiveStoreForTests();
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
});

afterEach(() => {
  vi.clearAllMocks();
  setActiveSelection(null);
  setRegisteredBackends([]);
  __resetActiveStoreForTests();
});

describe("CanvasExtensionIcon", () => {
  it("fetches the declared icon through the typed client and renders it as a blob URL", async () => {
    get.mockResolvedValue(svgIconBlob());

    render(
      <CanvasExtensionIcon extension={extensionWithIcon("assets/pulse.svg")} />,
    );

    const img = await screen.findByTestId("canvas-extension-icon");
    expect(img.getAttribute("src")).toMatch(/^blob:/);
    expect(get).toHaveBeenCalledWith(
      "/api/canvas-extensions/installed/demo-page/file?path=assets%2Fpulse.svg",
      { responseType: "blob" },
    );
    expect(
      screen.queryByTestId("canvas-extension-default-icon"),
    ).not.toBeInTheDocument();
  });

  it("revokes the object URL when the component unmounts", async () => {
    get.mockResolvedValue(svgIconBlob());
    const originalRevoke = URL.revokeObjectURL;
    const revokeSpy = vi.fn();
    URL.revokeObjectURL = revokeSpy as typeof URL.revokeObjectURL;

    try {
      const { unmount } = render(
        <CanvasExtensionIcon
          extension={extensionWithIcon("assets/pulse.svg")}
        />,
      );
      const img = await screen.findByTestId("canvas-extension-icon");
      const src = img.getAttribute("src");

      unmount();

      expect(revokeSpy).toHaveBeenCalledWith(src);
    } finally {
      URL.revokeObjectURL = originalRevoke;
    }
  });

  it("falls back to the default icon when no icon is declared", () => {
    render(<CanvasExtensionIcon extension={extensionWithIcon(null)} />);

    expect(
      screen.getByTestId("canvas-extension-default-icon"),
    ).toBeInTheDocument();
    expect(get).not.toHaveBeenCalled();
  });

  it("falls back to the default icon when the declared path is unsafe", () => {
    render(
      <CanvasExtensionIcon extension={extensionWithIcon("../../secret.svg")} />,
    );

    expect(
      screen.getByTestId("canvas-extension-default-icon"),
    ).toBeInTheDocument();
    expect(get).not.toHaveBeenCalled();
  });

  it("falls back to the default icon when the backend cannot serve the icon", async () => {
    get.mockRejectedValue(new Error("404"));

    render(
      <CanvasExtensionIcon extension={extensionWithIcon("assets/pulse.svg")} />,
    );

    const fallback = await screen.findByTestId("canvas-extension-default-icon");
    expect(fallback).toBeInTheDocument();
    expect(
      screen.queryByTestId("canvas-extension-icon"),
    ).not.toBeInTheDocument();
  });

  it("falls back to the default icon when the image errors after loading", async () => {
    get.mockResolvedValue(svgIconBlob());

    render(
      <CanvasExtensionIcon extension={extensionWithIcon("assets/pulse.svg")} />,
    );
    const img = await screen.findByTestId("canvas-extension-icon");

    fireEvent.error(img);

    expect(
      screen.getByTestId("canvas-extension-default-icon"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("canvas-extension-icon"),
    ).not.toBeInTheDocument();
  });
});
