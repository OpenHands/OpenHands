import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import type { InstalledCanvasExtensionInfo } from "#/types/canvas-extension";
import { CanvasExtensionIcon } from "./canvas-extension-icon";

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

describe("CanvasExtensionIcon", () => {
  it("renders the manifest-declared icon as an img", () => {
    render(
      <CanvasExtensionIcon extension={extensionWithIcon("assets/pulse.svg")} />,
    );

    const img = screen.getByTestId("canvas-extension-icon");
    expect(img).toHaveAttribute(
      "src",
      CanvasExtensionsService.buildIconUrl("demo-page", "assets/pulse.svg"),
    );
    expect(
      screen.queryByTestId("canvas-extension-default-icon"),
    ).not.toBeInTheDocument();
  });

  it("falls back to the default icon when no icon is declared", () => {
    render(<CanvasExtensionIcon extension={extensionWithIcon(null)} />);

    expect(
      screen.getByTestId("canvas-extension-default-icon"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("canvas-extension-icon"),
    ).not.toBeInTheDocument();
  });

  it("falls back to the default icon when the declared path is unsafe", () => {
    render(
      <CanvasExtensionIcon extension={extensionWithIcon("../../secret.svg")} />,
    );

    expect(
      screen.getByTestId("canvas-extension-default-icon"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("canvas-extension-icon"),
    ).not.toBeInTheDocument();
  });

  it("falls back to the default icon when the image fails to load", () => {
    render(
      <CanvasExtensionIcon extension={extensionWithIcon("assets/pulse.svg")} />,
    );

    fireEvent.error(screen.getByTestId("canvas-extension-icon"));

    expect(
      screen.getByTestId("canvas-extension-default-icon"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("canvas-extension-icon"),
    ).not.toBeInTheDocument();
  });
});
