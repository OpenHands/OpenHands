import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { CanvasExtensionHost } from "#/types/canvas-extension";

const FIXTURE_PATH = resolve(
  "src/fixtures/canvas-extensions/svg-edit/extension.js",
);

describe("SVG-edit Canvas Extension fixture", () => {
  afterEach(() => {
    document.body.replaceChildren();
  });

  it("registers the declared page and removes its iframe on cleanup", async () => {
    const source = await readFile(FIXTURE_PATH, "utf8");
    const moduleUrl = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
    const extensionModule = (await import(moduleUrl)) as {
      activate: (host: CanvasExtensionHost) => void | (() => void);
      SVG_EDIT_VERSION: string;
      SVG_EDIT_URL: string;
    };
    expect(extensionModule.SVG_EDIT_VERSION).toBe("7.4.2");
    expect(extensionModule.SVG_EDIT_URL).toBe(
      "https://unpkg.com/svgedit@7.4.2/dist/editor/index.html",
    );
    const container = document.createElement("div");
    document.body.append(container);
    const unregister = vi.fn();
    const registerPage = vi.fn<CanvasExtensionHost["registerPage"]>(
      () => unregister,
    );
    const host = {
      apiVersion: "1",
      extension: { name: "svg-edit", version: "0.1.0", resolvedRef: null },
      backend: { id: "local", kind: "local", orgId: null },
      registerPage,
      navigate: vi.fn(),
      agentServer: { request: vi.fn() },
    } satisfies CanvasExtensionHost;

    const disposeActivation = extensionModule.activate(host);
    expect(registerPage).toHaveBeenCalledWith("editor", expect.any(Function));

    const mountPage = registerPage.mock.calls[0]?.[1];
    expect(mountPage).toBeDefined();
    const disposePage = await mountPage?.({
      container,
      path: "",
      navigate: vi.fn(),
    });
    const frame = container.querySelector("iframe");
    expect(frame).toHaveAttribute(
      "src",
      "https://unpkg.com/svgedit@7.4.2/dist/editor/index.html",
    );
    expect(frame).toHaveAttribute("data-canvas-extension", "svg-edit");

    disposePage?.();
    expect(container).toBeEmptyDOMElement();

    disposeActivation?.();
    expect(unregister).toHaveBeenCalledOnce();
  });
});
