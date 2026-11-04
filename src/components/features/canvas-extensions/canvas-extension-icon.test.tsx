import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import type { InstalledCanvasExtensionInfo } from "#/types/canvas-extension";
import { CanvasExtensionIcon } from "./canvas-extension-icon";

function makeExtension(icon?: string): InstalledCanvasExtensionInfo {
  return {
    name: "demo-extension",
    version: "0.1.0",
    enabled: true,
    source: "github:example/demo",
    installed_at: "2026-08-01T00:00:00Z",
    install_path: "/tmp/demo-extension",
    manifest: {
      schema_version: 1,
      name: "demo-extension",
      version: "0.1.0",
      entrypoint: "dist/extension.js",
      icon,
    },
  };
}

function renderIcon(icon?: string) {
  const queryClient = new QueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <CanvasExtensionIcon extension={makeExtension(icon)} size={18} />
    </QueryClientProvider>,
  );
}

describe("CanvasExtensionIcon", () => {
  beforeEach(() => {
    URL.createObjectURL = vi.fn(() => "blob:icon");
    URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the default icon without fetching when no icon is declared", () => {
    const fetchIcon = vi.spyOn(CanvasExtensionsService, "fetchIcon");

    renderIcon();

    expect(
      screen.getByTestId("canvas-extension-default-icon"),
    ).toBeInTheDocument();
    expect(fetchIcon).not.toHaveBeenCalled();
  });

  it("renders the fetched icon and revokes its object URL on unmount", async () => {
    vi.spyOn(CanvasExtensionsService, "fetchIcon").mockResolvedValue(
      new Blob(["<svg/>"], { type: "image/svg+xml" }),
    );

    const { unmount } = renderIcon("assets/icon.svg");

    expect(await screen.findByTestId("canvas-extension-icon")).toHaveAttribute(
      "src",
      "blob:icon",
    );
    expect(CanvasExtensionsService.fetchIcon).toHaveBeenCalledWith(
      "demo-extension",
    );
    unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:icon");
  });

  it("keeps the default icon when the backend cannot serve the icon", async () => {
    vi.spyOn(CanvasExtensionsService, "fetchIcon").mockRejectedValue(
      new Error("404"),
    );

    renderIcon("assets/icon.svg");

    await vi.waitFor(() =>
      expect(CanvasExtensionsService.fetchIcon).toHaveBeenCalled(),
    );
    expect(
      screen.getByTestId("canvas-extension-default-icon"),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("canvas-extension-icon")).toBeNull();
  });
});
