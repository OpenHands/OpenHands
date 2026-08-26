import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { CanvasExtensionHost } from "#/types/canvas-extension";

const FIXTURE_PATH = resolve(
  "tests/fixtures/canvas-extensions/svg-edit/extension.js",
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
    };
    const container = document.createElement("div");
    document.body.append(container);
    let mountPage: Parameters<CanvasExtensionHost["registerPage"]>[1] | null =
      null;
    const unregister = vi.fn();
    const host = {
      apiVersion: "1",
      extension: { name: "svg-edit", version: "0.1.0", resolvedRef: null },
      backend: { id: "local", kind: "local", orgId: null },
      registerPage: vi.fn((_id, mount) => {
        mountPage = mount;
        return unregister;
      }),
      navigate: vi.fn(),
      agentServer: { request: vi.fn() },
    } satisfies CanvasExtensionHost;

    const disposeActivation = extensionModule.activate(host);
    expect(host.registerPage).toHaveBeenCalledWith(
      "editor",
      expect.any(Function),
    );

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
