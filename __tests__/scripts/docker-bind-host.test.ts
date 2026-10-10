// @vitest-environment node

import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const entrypoint = readFileSync(
  path.join(repoRoot, "docker/entrypoint.sh"),
  "utf-8",
);
const blockStart = "# >>> docker-bind-host";
const blockEnd = "# <<< docker-bind-host";

function dockerBindHostBlock(): string {
  const start = entrypoint.indexOf(blockStart);
  const end = entrypoint.indexOf(blockEnd);
  if (start === -1 || end === -1) {
    throw new Error("Docker bind-host markers are missing");
  }
  return entrypoint.slice(start, end);
}

function resolveDockerBindHost(options: {
  ipv6ProbeFile?: string;
  bindHostEnv?: string;
}): string {
  const script = [
    "set -uo pipefail",
    dockerBindHostBlock(),
    'printf "%s" "$STATIC_SERVER_HOST"',
  ].join("\n");
  const env: Record<string, string> = { PATH: process.env.PATH ?? "" };
  if (options.ipv6ProbeFile !== undefined) {
    env.OH_IPV6_PROBE_FILE = options.ipv6ProbeFile;
  }
  if (options.bindHostEnv !== undefined) {
    env.OH_BIND_HOST = options.bindHostEnv;
  }
  const result = spawnSync("bash", ["-c", script], {
    encoding: "utf-8",
    env,
  });
  expect(result.status).toBe(0);
  return result.stdout.trim();
}

describe("Docker static server bind host resolution", () => {
  it("does not hardcode --host :: in static server invocations", () => {
    // Neither the main nor public-mode static-server should use a hardcoded --host ::
    expect(entrypoint).not.toMatch(/static-server\.mjs[\s\S]*?--host ::/);
  });

  it("binds :: on dual-stack hosts where IPv6 probe file exists", () => {
    const host = resolveDockerBindHost({ ipv6ProbeFile: "/dev/null" });
    expect(host).toBe("::");
  });

  it("falls back to 0.0.0.0 on IPv4-only hosts where IPv6 probe file is absent", () => {
    const host = resolveDockerBindHost({
      ipv6ProbeFile: "/nonexistent/proc/net/if_inet6",
    });
    expect(host).toBe("0.0.0.0");
  });

  it("honors explicit OH_BIND_HOST override", () => {
    const host = resolveDockerBindHost({
      bindHostEnv: "127.0.0.1",
      ipv6ProbeFile: "/dev/null",
    });
    expect(host).toBe("127.0.0.1");
  });
});
