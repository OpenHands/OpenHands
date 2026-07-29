// @vitest-environment node
// The afterPack hook manipulates real directories and its contract is that
// the packaged app's spawned servers can resolve their bare npm imports
// (sirv, httpxy) OUTSIDE a repo checkout — Node's ESM resolution walks up
// from the importing file, so a bundle tested inside the repo silently
// resolves against the repo's node_modules and hides a missing package.
// These tests therefore build a fake bundle under os.tmpdir() and verify
// resolution with a real `node --eval` from that location.
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import config from "../../electron-builder.config.mjs";

const afterPack = config.afterPack as (ctx: unknown) => Promise<void>;

const PRODUCT_FILENAME = "Agent Canvas";

function makeContext(platform: string, appOutDir: string) {
  return {
    electronPlatformName: platform,
    appOutDir,
    packager: { appInfo: { productFilename: PRODUCT_FILENAME } },
  };
}

/** Resolve the packaged app dir for a platform, mirroring the bundle layout. */
function appDirFor(platform: string, appOutDir: string) {
  return platform === "darwin"
    ? join(appOutDir, `${PRODUCT_FILENAME}.app`, "Contents", "Resources", "app")
    : join(appOutDir, "resources", "app");
}

/**
 * Run `await import("sirv"); await import("httpxy")` with a real Node
 * process whose cwd is the packaged app dir — exactly how the spawned
 * static-server/ingress scripts resolve their imports at runtime.
 */
function resolveRuntimeImports(appDir: string) {
  return spawnSync(
    process.execPath,
    ["--input-type=module", "-e", 'await import("sirv"); await import("httpxy");'],
    { cwd: appDir, stdio: "pipe" },
  );
}

describe("electron-builder afterPack hook", () => {
  let tmp: string | null = null;

  afterEach(() => {
    if (tmp) rmSync(tmp, { recursive: true, force: true });
    tmp = null;
  });

  it("strips the auto-bundled node_modules and restores a resolvable runtime closure (macOS layout)", async () => {
    tmp = mkdtempSync(join(tmpdir(), "eb-afterpack-"));
    const appDir = appDirFor("darwin", tmp);
    // Simulate electron-builder's accidental copy of the hoisted dev tree.
    const junkPkg = join(appDir, "node_modules", "react");
    mkdirSync(junkPkg, { recursive: true });
    writeFileSync(join(junkPkg, "package.json"), '{"name":"react"}');

    await afterPack(makeContext("darwin", tmp));

    expect(existsSync(junkPkg)).toBe(false);
    const result = resolveRuntimeImports(appDir);
    expect(result.status, String(result.stderr)).toBe(0);
  });

  it("restores the runtime closure even when no node_modules was bundled", async () => {
    // If electron-builder ever stops copying the hoisted tree, the restore
    // must still run — otherwise the installed app regresses to
    // ERR_MODULE_NOT_FOUND in ingress/static-server.
    tmp = mkdtempSync(join(tmpdir(), "eb-afterpack-"));
    const appDir = appDirFor("darwin", tmp);
    mkdirSync(appDir, { recursive: true });

    await afterPack(makeContext("darwin", tmp));

    const result = resolveRuntimeImports(appDir);
    expect(result.status, String(result.stderr)).toBe(0);
  });

  it("operates on the flat resources/app layout for non-mac platforms", async () => {
    tmp = mkdtempSync(join(tmpdir(), "eb-afterpack-"));
    const appDir = appDirFor("linux", tmp);
    const junkPkg = join(appDir, "node_modules", "react");
    mkdirSync(junkPkg, { recursive: true });
    writeFileSync(join(junkPkg, "package.json"), '{"name":"react"}');

    await afterPack(makeContext("linux", tmp));

    expect(existsSync(junkPkg)).toBe(false);
    expect(existsSync(join(appDir, "node_modules", "sirv", "package.json"))).toBe(true);
    expect(existsSync(join(appDir, "node_modules", "httpxy", "package.json"))).toBe(true);
  });
});
