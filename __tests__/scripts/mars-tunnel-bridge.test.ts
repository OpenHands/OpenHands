// @vitest-environment node
import { describe, expect, it, vi } from "vitest";

import {
  MARS_TUNNEL_IPC,
  createMarsTunnelBridge,
} from "../../scripts/mars-tunnel-bridge.mjs";

/** A fake ipcMain that just records handlers so tests can invoke them directly. */
function fakeIpcMain() {
  const handlers = new Map<string, (event: unknown, ...args: unknown[]) => unknown>();
  return {
    handle: vi.fn((channel: string, fn: (event: unknown, ...args: unknown[]) => unknown) => {
      handlers.set(channel, fn);
    }),
    invoke: (channel: string, ...args: unknown[]) => {
      const handler = handlers.get(channel);
      if (!handler) throw new Error(`no handler registered for ${channel}`);
      return handler({}, ...args);
    },
  };
}

function fakeRegistry() {
  return {
    attach: vi.fn(async (params) => ({ ...params, localPort: 51000, status: "connected" })),
    detach: vi.fn(async () => {}),
    detachAll: vi.fn(async () => {}),
    get: vi.fn(() => undefined),
    list: vi.fn(() => []),
  };
}

describe("createMarsTunnelBridge", () => {
  it("registers handlers for openTunnel, closeTunnel, and getTunnel", () => {
    const registry = fakeRegistry();
    const bridge = createMarsTunnelBridge({ registry });
    const ipcMain = fakeIpcMain();

    bridge.registerIpc(ipcMain);

    expect(ipcMain.handle).toHaveBeenCalledWith(MARS_TUNNEL_IPC.openTunnel, expect.any(Function));
    expect(ipcMain.handle).toHaveBeenCalledWith(MARS_TUNNEL_IPC.closeTunnel, expect.any(Function));
    expect(ipcMain.handle).toHaveBeenCalledWith(MARS_TUNNEL_IPC.getTunnel, expect.any(Function));
  });

  it("openTunnel forwards its params straight to registry.attach", async () => {
    const registry = fakeRegistry();
    const bridge = createMarsTunnelBridge({ registry });
    const ipcMain = fakeIpcMain();
    bridge.registerIpc(ipcMain);

    const params = { sessionId: "sess_a", remotePort: 8000, accessToken: "t" };
    const result = await ipcMain.invoke(MARS_TUNNEL_IPC.openTunnel, params);

    expect(registry.attach).toHaveBeenCalledWith(params);
    expect(result).toMatchObject({ sessionId: "sess_a", localPort: 51000 });
  });

  it("closeTunnel forwards the sessionId to registry.detach", async () => {
    const registry = fakeRegistry();
    const bridge = createMarsTunnelBridge({ registry });
    const ipcMain = fakeIpcMain();
    bridge.registerIpc(ipcMain);

    await ipcMain.invoke(MARS_TUNNEL_IPC.closeTunnel, "sess_a");

    expect(registry.detach).toHaveBeenCalledWith("sess_a");
  });

  it("getTunnel forwards the sessionId to registry.get", async () => {
    const registry = fakeRegistry();
    const bridge = createMarsTunnelBridge({ registry });
    const ipcMain = fakeIpcMain();
    bridge.registerIpc(ipcMain);

    await ipcMain.invoke(MARS_TUNNEL_IPC.getTunnel, "sess_a");

    expect(registry.get).toHaveBeenCalledWith("sess_a");
  });

  it("dispose() tears down every tunnel via registry.detachAll", async () => {
    const registry = fakeRegistry();
    const bridge = createMarsTunnelBridge({ registry });

    await bridge.dispose();

    expect(registry.detachAll).toHaveBeenCalledTimes(1);
  });

  it("defaults to a real tunnel registry when none is provided", () => {
    const bridge = createMarsTunnelBridge();
    expect(bridge.registry).toBeTruthy();
    expect(typeof bridge.registry.attach).toBe("function");
  });
});
