// @vitest-environment node
import { spawn } from "node:child_process";
import net from "node:net";
import { describe, expect, it } from "vitest";

import {
  buildPortForwardArgs,
  parseForwardedPort,
  resolveDoctlBinary,
  startPortForwardTunnel,
} from "../../scripts/tunnel-client.mjs";

describe("resolveDoctlBinary", () => {
  it("resolves to the bundled binary under resourcesPath/bin when packaged", () => {
    expect(
      resolveDoctlBinary({
        resourcesPath: "/Applications/Foo.app/Contents/Resources",
        isPackaged: true,
        platform: "darwin",
      }),
    ).toBe("/Applications/Foo.app/Contents/Resources/bin/doctl");
  });

  it("appends .exe on Windows", () => {
    // node:path's join() is not platform-emulated by the `platform` option —
    // like main.mjs's injectBundledUv, this only ever runs with the real
    // path separator of the host OS, so the test only asserts the binary
    // name switches to doctl.exe.
    expect(
      resolveDoctlBinary({
        resourcesPath: "/opt/foo/resources",
        isPackaged: true,
        platform: "win32",
      }),
    ).toBe("/opt/foo/resources/bin/doctl.exe");
  });

  it("falls back to a bare PATH lookup outside a packaged app", () => {
    expect(resolveDoctlBinary({ isPackaged: false, platform: "darwin" })).toBe(
      "doctl",
    );
    expect(resolveDoctlBinary({ isPackaged: true, platform: "linux" })).toBe(
      "doctl",
    );
  });
});

describe("buildPortForwardArgs", () => {
  it("builds the agents port-forward argv with 0 letting doctl pick the local port", () => {
    expect(
      buildPortForwardArgs({ sessionId: "sess_abc123", remotePort: 8000 }),
    ).toEqual(["agents", "port-forward", "sess_abc123", "0:8000"]);
  });

  it("honors an explicit local port", () => {
    expect(
      buildPortForwardArgs({
        sessionId: "sess_abc123",
        remotePort: 8000,
        localPort: 54321,
      }),
    ).toEqual(["agents", "port-forward", "sess_abc123", "54321:8000"]);
  });

  it("rejects a missing session id", () => {
    expect(() =>
      buildPortForwardArgs({ sessionId: "", remotePort: 8000 }),
    ).toThrow(/sessionId/);
  });

  it("rejects an out-of-range remote port", () => {
    expect(() =>
      buildPortForwardArgs({ sessionId: "sess_abc123", remotePort: 0 }),
    ).toThrow(/Invalid remote port/);
    expect(() =>
      buildPortForwardArgs({ sessionId: "sess_abc123", remotePort: 70000 }),
    ).toThrow(/Invalid remote port/);
  });
});

describe("parseForwardedPort", () => {
  it("parses doctl's forwarding announcement", () => {
    expect(
      parseForwardedPort(
        "Forwarding 127.0.0.1:54321 -> port 8000 in session sess_abc123",
      ),
    ).toEqual({
      address: "127.0.0.1",
      localPort: 54321,
      remotePort: 8000,
      sessionId: "sess_abc123",
    });
  });

  it("returns null for unrelated output", () => {
    expect(parseForwardedPort("Ready. Press Ctrl-C to stop.")).toBeNull();
    expect(parseForwardedPort("")).toBeNull();
  });
});

/**
 * Build a spawnFn that runs a small Node script standing in for `doctl
 * agents port-forward`: it opens a real TCP listener, announces it exactly
 * like doctl does, then idles until killed. This exercises the real
 * stdout-parsing + TCP-health-check + process-lifecycle path without
 * depending on an actual doctl binary.
 */
function fakeDoctlSpawnFn({ sessionId = "sess_abc123", remotePort = 8000 } = {}) {
  const script = `
    const net = require("node:net");
    const server = net.createServer((socket) => socket.end());
    server.listen(0, "127.0.0.1", () => {
      const port = server.address().port;
      console.log(\`Forwarding 127.0.0.1:\${port} -> port ${remotePort} in session ${sessionId}\`);
      console.log("Ready. Press Ctrl-C to stop.");
    });
    process.on("SIGTERM", () => process.exit(0));
  `;
  return (_command, _args, options) =>
    spawn(process.execPath, ["-e", script], options);
}

function fakeFailingDoctlSpawnFn(stderrMessage) {
  const script = `
    console.error(${JSON.stringify(stderrMessage)});
    process.exit(1);
  `;
  return (_command, _args, options) =>
    spawn(process.execPath, ["-e", script], options);
}

describe("startPortForwardTunnel", () => {
  it("resolves once doctl announces a forwarded port and the listener is healthy", async () => {
    const tunnel = await startPortForwardTunnel({
      sessionId: "sess_abc123",
      remotePort: 8000,
      accessToken: "test-token",
      spawnFn: fakeDoctlSpawnFn({ sessionId: "sess_abc123", remotePort: 8000 }),
    });

    try {
      expect(tunnel.sessionId).toBe("sess_abc123");
      expect(tunnel.remotePort).toBe(8000);
      expect(tunnel.localPort).toBeGreaterThan(0);

      // The resolved local port really is connectable.
      await new Promise<void>((resolve, reject) => {
        const socket = net.connect(
          { port: tunnel.localPort, host: "127.0.0.1" },
          () => {
            socket.end();
            resolve();
          },
        );
        socket.once("error", reject);
      });
    } finally {
      tunnel.stop();
    }
  });

  it("rejects when accessToken is missing", async () => {
    await expect(
      startPortForwardTunnel({ sessionId: "sess_abc123", remotePort: 8000 }),
    ).rejects.toThrow(/accessToken/);
  });

  it("rejects with the doctl stderr tail when the process exits before announcing a port", async () => {
    await expect(
      startPortForwardTunnel({
        sessionId: "sess_abc123",
        remotePort: 8000,
        accessToken: "test-token",
        spawnFn: fakeFailingDoctlSpawnFn("server rejected tunnel (403 Forbidden): invalid token"),
      }),
    ).rejects.toThrow(/invalid token/);
  });
});
