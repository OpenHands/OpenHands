// @vitest-environment node
// The afterPack hook manipulates real directories and its contract is that
// the packaged app's spawned servers can resolve their bare npm imports
// (sirv, httpxy) OUTSIDE a repo checkout — Node's ESM resolution walks up
// from the importing file, so a bundle tested inside the repo silently
// resolves against the repo's node_modules and hides a missing package.
// These tests therefore build a fake bundle under os.tmpdir() and verify
// resolution with a real `node --eval` from that location.
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import config from "../../electron-builder.config.mjs";

const afterPack = config.afterPack as (ctx: unknown) => Promise<void>;

const PRODUCT_FILENAME = "OpenHands Agent Canvas";

function makeContext(platform: string, appOutDir: string) {
  return {
    electronPlatformName: platform,
    appOutDir,
    packager: { appInfo: { productFilename: PRODUCT_FILENAME } },
  };
}

/** Resolve the packaged Resources dir for a platform, mirroring the bundle layout. */
function resourcesDirFor(platform: string, appOutDir: string) {
  return platform === "darwin"
    ? join(appOutDir, `${PRODUCT_FILENAME}.app`, "Contents", "Resources")
    : join(appOutDir, "resources");
}

/** Resolve the packaged app dir for a platform, mirroring the bundle layout. */
function appDirFor(platform: string, appOutDir: string) {
  return join(resourcesDirFor(platform, appOutDir), "app");
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

/**
 * Write a fake POSIX-layout bundled Node dir (npm at lib/node_modules) with
 * its npm-cli.js marker, as download-node.mjs produces on macOS/Linux.
 */
function writeFakeNpmCli(resourcesDir: string, nodeDirName: string) {
  const npmCli = join(resourcesDir, nodeDirName, "lib", "node_modules", "npm", "bin", "npm-cli.js");
  mkdirSync(dirname(npmCli), { recursive: true });
  writeFileSync(npmCli, "// fake npm-cli.js\n");
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

  it("verifies npm in every per-arch node dir of a universal bundle", async () => {
    tmp = mkdtempSync(join(tmpdir(), "eb-afterpack-"));
    const resourcesDir = resourcesDirFor("darwin", tmp);
    mkdirSync(join(resourcesDir, "app"), { recursive: true });
    writeFakeNpmCli(resourcesDir, "node-arm64");
    writeFakeNpmCli(resourcesDir, "node-x64");

    await expect(afterPack(makeContext("darwin", tmp))).resolves.toBeUndefined();
  });

  it("fails the build when one per-arch node dir of a universal bundle is missing npm", async () => {
    tmp = mkdtempSync(join(tmpdir(), "eb-afterpack-"));
    const resourcesDir = resourcesDirFor("darwin", tmp);
    mkdirSync(join(resourcesDir, "app"), { recursive: true });
    writeFakeNpmCli(resourcesDir, "node-arm64");
    // x64 slice gutted: the dir exists but its npm-cli.js does not — the
    // hook must verify BOTH slices, not just the first one it finds.
    mkdirSync(join(resourcesDir, "node-x64"), { recursive: true });

    await expect(afterPack(makeContext("darwin", tmp))).rejects.toThrow(
      /missing npm-cli\.js at .*node-x64/,
    );
  });

  it("still verifies the legacy flat node dir (single-arch back-compat)", async () => {
    tmp = mkdtempSync(join(tmpdir(), "eb-afterpack-"));
    const resourcesDir = resourcesDirFor("darwin", tmp);
    mkdirSync(join(resourcesDir, "app"), { recursive: true });
    writeFakeNpmCli(resourcesDir, "node");

    await expect(afterPack(makeContext("darwin", tmp))).resolves.toBeUndefined();
  });
});

// The config is a .mjs module whose extraResources is computed at evaluation
// time from ELECTRON_ARCH (build:desktop:universal sets it before invoking
// electron-builder). Each case therefore re-imports the module fresh, with
// the env var scrubbed in beforeEach and restored in afterEach.
const ELECTRON_ARCH_ENV = "ELECTRON_ARCH";

describe("extraResources mode switch", () => {
  let savedArch: string | undefined;

  async function importConfig() {
    vi.resetModules();
    return (await import("../../electron-builder.config.mjs")).default;
  }

  beforeEach(() => {
    savedArch = process.env[ELECTRON_ARCH_ENV];
    delete process.env[ELECTRON_ARCH_ENV];
  });

  afterEach(() => {
    if (savedArch === undefined) delete process.env[ELECTRON_ARCH_ENV];
    else process.env[ELECTRON_ARCH_ENV] = savedArch;
  });

  it("ships the legacy flat runtime dirs when ELECTRON_ARCH is unset", async () => {
    const cfg = await importConfig();

    expect(cfg.extraResources).toEqual([
      { from: "resources/bin/", to: "bin/", filter: ["**/*"] },
      { from: "resources/node/", to: "node/", filter: ["**/*"] },
    ]);
  });

  it("ships both per-arch runtime dirs into every universal pass, whitelisted for the merger", async () => {
    process.env[ELECTRON_ARCH_ENV] = "universal";
    const cfg = await importConfig();

    // Both per-arch dirs go into BOTH packaging passes: @electron/universal
    // requires the x64 and arm64 pass trees to carry the same Mach-O file
    // list, and mac.x64ArchFiles whitelists our intentionally identical
    // per-arch runtime binaries through the merge. Full rationale in
    // electron-builder.config.mjs.
    expect(cfg.extraResources).toEqual([
      { from: "resources/bin-arm64/", to: "bin-arm64/", filter: ["**/*"] },
      { from: "resources/bin-x64/", to: "bin-x64/", filter: ["**/*"] },
      { from: "resources/node-arm64/", to: "node-arm64/", filter: ["**/*"] },
      { from: "resources/node-x64/", to: "node-x64/", filter: ["**/*"] },
    ]);
    expect(cfg.mac).toBeDefined();
    expect(cfg.mac!.x64ArchFiles).toBe("**/{bin,node}-{arm64,x64}/**");
  });
});

// The app's display name is set in two places that must agree:
//   electron-builder.config.mjs productName → CFBundleName / NSIS / .desktop
//     name of the PACKAGED bundle (what the Dock, ⌘-Tab and Finder show).
//   electron/package.json productName       → app.name at runtime (macOS menu
//     bar, About panel, app.getPath("userData")), read by Electron's
//     default_app in dev and by lib/browser/init when packaged.
// Neither one implies the other — the packaged app shipped with
// CFBundleName "Agent Canvas" but app.name "agent-canvas" until both were
// set. Pin them together so they can't drift again.
describe("desktop app name", () => {
  it("keeps electron/package.json productName in sync with the builder config", () => {
    const appManifest = JSON.parse(
      readFileSync(join(import.meta.dirname, "../../electron/package.json"), "utf8"),
    );

    expect(appManifest.productName).toBe(config.productName);
    expect(config.productName).toBe(PRODUCT_FILENAME);
  });
});
