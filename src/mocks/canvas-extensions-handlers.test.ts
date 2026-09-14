/* eslint-disable local/no-direct-agent-server-fetch --
   These fetches deliberately bypass the typed client to exercise the MSW
   network-interception layer that dev:mock relies on. */
import { afterEach, describe, expect, it } from "vitest";
import {
  CANVAS_EXTENSION_DEMO_SOURCE,
  resetCanvasExtensionsMockData,
} from "#/mocks/canvas-extensions-handlers";

afterEach(() => {
  resetCanvasExtensionsMockData();
});

async function installDemo(): Promise<Response> {
  return fetch("http://127.0.0.1:8000/api/canvas-extensions/install", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source: CANVAS_EXTENSION_DEMO_SOURCE }),
  });
}

describe("Canvas Extension MSW handlers", () => {
  it("installs the fixture disabled and then enables it explicitly", async () => {
    const installResponse = await installDemo();
    expect(installResponse.status).toBe(200);
    await expect(installResponse.json()).resolves.toMatchObject({
      name: "demo-page",
      enabled: false,
      source: CANVAS_EXTENSION_DEMO_SOURCE,
    });

    const enableResponse = await fetch(
      "http://127.0.0.1:8000/api/canvas-extensions/installed/demo-page",
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: true }),
      },
    );
    await expect(enableResponse.json()).resolves.toEqual({
      name: "demo-page",
      enabled: true,
    });

    const listResponse = await fetch(
      "http://127.0.0.1:8000/api/canvas-extensions/installed",
    );
    await expect(listResponse.json()).resolves.toMatchObject({
      canvas_extensions: [{ name: "demo-page", enabled: true }],
    });
  });

  it("serves the fixture as JavaScript text", async () => {
    await installDemo();

    const response = await fetch(
      "http://127.0.0.1:8000/api/canvas-extensions/installed/demo-page/bundle",
    );

    expect(response.headers.get("content-type")).toContain(
      "application/javascript",
    );
    await expect(response.text()).resolves.toContain(
      "export function activate",
    );
  });

  it("rejects arbitrary sources in mock mode", async () => {
    const response = await fetch(
      "http://127.0.0.1:8000/api/canvas-extensions/install",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: "github:example/unknown" }),
      },
    );

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      detail: `Mock mode only installs '${CANVAS_EXTENSION_DEMO_SOURCE}'.`,
    });
  });
});
