#!/usr/bin/env node
/**
 * Download the official Node.js distribution for the current platform into
 * resources/node/, so electron-builder can bundle it as an extraResource.
 *
 * The packaged Electron desktop app uses this bundled Node to provide
 * `node`, `npm`, and `npx` to spawned subprocesses — most importantly the
 * stdio MCP servers in the marketplace (Slack, GitHub, Figma, etc.) whose
 * commands start with `npx -y <package>`.
 *
 * Why bundle Node instead of using Electron-as-Node (ELECTRON_RUN_AS_NODE=1)?
 *
 *   We tried that first. Electron-as-Node works fine for our backend
 *   helper scripts (static-server.mjs, ingress.mjs) which mostly do
 *   networking, but it is **not** reliable for stdio JSON-RPC servers.
 *   When npx-cli.js (running under Electron-as-Node) spawned the MCP
 *   server, the child's stdin pipe semantics differed from vanilla Node
 *   on macOS (the parent is a windowed Electron process, not a clean
 *   command-line Node binary) — the server appeared to start, then
 *   immediately exited with "McpError: Connection closed" before the
 *   first JSON-RPC handshake message could land. Bundling the real
 *   Node binary sidesteps all of that.
 *
 * The downloaded Node.js distribution already includes npm and npx at
 * `bin/npm` / `bin/npx` (POSIX) or `npm.cmd` / `npx.cmd` (Windows), so we
 * do **not** need a separate npm download (this script supersedes the
 * earlier download-npm.mjs).
 *
 * Usage:
 *   node scripts/download-node.mjs           # uses NODE_BUNDLE_VERSION below
 *   NODE_VERSION=22.10.0 node scripts/download-node.mjs
 *
 * Output (per platform):
 *   POSIX:   resources/node/bin/{node,npm,npx} + resources/node/lib/node_modules/npm/...
 *   Windows: resources/node/{node.exe,npm.cmd,npx.cmd} + resources/node/node_modules/npm/...
 */

import {
  chmodSync,
  createWriteStream,
  existsSync,
  lstatSync,
  mkdirSync,
  readdirSync,
  readlinkSync,
  rmSync,
  statSync,
  unlinkSync,
} from "node:fs";
import { get } from "node:https";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, "..");
const outDir = join(projectRoot, "resources", "node");

// Pinned Node version. Electron 42 ships Node 22, so we bundle a 22.x
// LTS release to match the embedded runtime's ABI/native-module surface.
// We intentionally use 22.12.0 — the repo's own support floor
// (package.json engines.node >=22.12.0, volta 22.12.0) — rather than
// Electron 42.3.2's exact embedded Node patch level: the bundled binary
// runs this repo's launcher scripts, and native modules only need ABI
// parity (NODE_MODULE_VERSION 127, shared by all 22.x builds).
// Override at build time with NODE_VERSION=… (e.g. to test against a
// newer release). Major version >=22 only; engines.node in npm 10.x
// requires ^18.17.0 || >=20.5.0.
const NODE_BUNDLE_VERSION = "22.12.0";

// ── Platform detection ───────────────────────────────────────────────────────

const PLATFORM = process.platform; // 'darwin' | 'linux' | 'win32'
const ARCH = process.arch; // 'x64' | 'arm64' | 'ia32'

/**
 * Map (platform, arch) → Node's published distribution name.
 * Names come straight from https://nodejs.org/dist/<version>/.
 *
 *   macOS arm64  → node-v<ver>-darwin-arm64.tar.gz
 *   macOS x64    → node-v<ver>-darwin-x64.tar.gz
 *   linux x64    → node-v<ver>-linux-x64.tar.gz
 *   linux arm64  → node-v<ver>-linux-arm64.tar.gz
 *   win32 x64    → node-v<ver>-win-x64.zip
 *   win32 arm64  → node-v<ver>-win-arm64.zip
 */
function getPlatformSpec(version) {
  const base = `node-v${version}`;
  if (PLATFORM === "darwin") {
    const arch = ARCH === "arm64" ? "arm64" : "x64";
    return { name: `${base}-darwin-${arch}`, ext: "tar.gz" };
  }
  if (PLATFORM === "linux") {
    const arch = ARCH === "arm64" ? "arm64" : "x64";
    return { name: `${base}-linux-${arch}`, ext: "tar.gz" };
  }
  if (PLATFORM === "win32") {
    const arch = ARCH === "arm64" ? "arm64" : "x64";
    return { name: `${base}-win-${arch}`, ext: "zip" };
  }
  throw new Error(`Unsupported platform for Node download: ${PLATFORM}/${ARCH}`);
}

// ── Version resolution ───────────────────────────────────────────────────────

function resolveVersion() {
  const requested = process.env.NODE_VERSION?.replace(/^v/, "");
  return requested || NODE_BUNDLE_VERSION;
}

// ── HTTP helpers ─────────────────────────────────────────────────────────────

function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const file = createWriteStream(dest);
    function doGet(u) {
      get(u, { headers: { "User-Agent": "agent-canvas-build" } }, (res) => {
        if (res.statusCode === 301 || res.statusCode === 302) {
          return doGet(res.headers.location);
        }
        if (res.statusCode !== 200) {
          file.destroy();
          return reject(new Error(`GET ${u} → HTTP ${res.statusCode}`));
        }
        res.pipe(file);
        file.on("finish", () => file.close(resolve));
        file.on("error", reject);
        res.on("error", reject);
      }).on("error", (err) => {
        file.destroy();
        reject(err);
      });
    }
    doGet(url);
  });
}

// ── Extraction ───────────────────────────────────────────────────────────────

function extract(archivePath, targetDir, ext) {
  // Both tar.gz and zip extract via system `tar`:
  //   GNU/BSD tar (macOS/Linux) handles .tar.gz natively.
  //   bsdtar (Windows 10+) handles both .tar.gz and .zip.
  // --strip-components=1 drops the "node-vX.Y.Z-<platform>-<arch>/" top dir.
  void ext; // archive content is identified by tar's own magic bytes
  execFileSync(
    "tar",
    ["-xf", archivePath, "-C", targetDir, "--strip-components=1"],
    { stdio: "inherit" },
  );
}

function ensureExecutable(p) {
  if (process.platform === "win32") return;
  try {
    chmodSync(p, 0o755);
  } catch {}
}

// ── Layout verification ──────────────────────────────────────────────────────

/**
 * Confirm the extracted tree has the binaries we depend on.
 * On Unix Node puts them in bin/; on Windows they live at the root.
 */
function verifyLayout() {
  const isWin = PLATFORM === "win32";
  const required = isWin
    ? ["node.exe", "npm.cmd", "npx.cmd"]
    : ["bin/node", "bin/npm", "bin/npx"];
  for (const rel of required) {
    const p = join(outDir, rel);
    if (!existsSync(p)) {
      throw new Error(
        `Expected ${rel} in extracted Node distribution but it is missing ` +
          `at ${p}. Did the tarball layout change?`,
      );
    }
    ensureExecutable(p);
  }

  // npm/npx are wrapper scripts that invoke node against npm's JS entry
  // points; verify the targets exist too so a packaged build doesn't ship
  // a half-broken installation.
  const npmCli = isWin
    ? join(outDir, "node_modules", "npm", "bin", "npm-cli.js")
    : join(outDir, "lib", "node_modules", "npm", "bin", "npm-cli.js");
  if (!existsSync(npmCli)) {
    throw new Error(`Bundled Node is missing npm-cli.js at ${npmCli}`);
  }
}

// ── Pruning ──────────────────────────────────────────────────────────────────

/**
 * Drop pieces of the Node distribution that are only useful when building
 * native modules from source or for human-readable documentation. Stripping
 * these shrinks the bundled Node from ~170 MB → ~115 MB on Linux x64 (the
 * Node binary itself is the bulk of what remains and can't be reduced).
 *
 * Kept intentionally:
 *   bin/node, bin/npm, bin/npx      — runtime binaries / wrappers
 *   lib/node_modules/{npm,corepack} — npm itself
 *   LICENSE                         — required by the BSD-style Node license
 */
function pruneUnusedFiles() {
  // IMPORTANT: every entry that points into a directory we delete must also
  // delete any symlink/shim that targets into it, otherwise electron-builder
  // hits ENOENT trying to stat() the dangling symlink while copying the
  // extraResource into the .app bundle.
  //
  // Example: Node's POSIX tarball ships `bin/corepack` as a symlink to
  // `../lib/node_modules/corepack/dist/corepack.js`. If we drop the corepack
  // module under lib/ but leave the symlink, `electron-builder` fails with
  //   ENOENT: ... Resources/node/bin/corepack
  const candidates =
    PLATFORM === "win32"
      ? [
          // Windows Node zip lays out npm directly under node_modules/, not lib/.
          // Strip docs, headers, and node_modules/corepack (the npm runtime
          // doesn't need corepack to run, and we don't ship yarn/pnpm).
          "CHANGELOG.md",
          "README.md",
          "node_modules/corepack",
          // Windows ships corepack as both a Bash wrapper and a cmd.exe wrapper
          // at the distribution root; both proxy into node_modules/corepack.
          "corepack",
          "corepack.cmd",
        ]
      : [
          // POSIX layout — keep bin/ and lib/node_modules/npm; drop the rest.
          "include",
          "share",
          "CHANGELOG.md",
          "README.md",
          "lib/node_modules/corepack",
          // Symlink in bin/ targets the corepack we just deleted.
          "bin/corepack",
        ];
  for (const rel of candidates) {
    const p = join(outDir, rel);
    // `rmSync(force: true)` resolves the path through symlinks, so once the
    // corepack target directory is deleted the now-dangling `bin/corepack`
    // link reads as "already gone" and silently survives — the exact ENOENT
    // trap failOnDanglingSymlinks() exists to catch. Remove files/symlinks
    // with `unlinkSync` (lstat semantics, works on dangling links) first and
    // fall back to `rmSync` for directories.
    try {
      unlinkSync(p);
    } catch {
      try {
        rmSync(p, { recursive: true, force: true });
      } catch {
        // best-effort
      }
    }
  }
}

// ── Dangling-symlink check ───────────────────────────────────────────────────

/**
 * Walk the pruned tree and refuse to finish if any symlink points at a path
 * that no longer exists. electron-builder calls `stat()` (which follows
 * symlinks) on every entry it copies into the .app bundle, so a single
 * dangling symlink blows up the whole `build:desktop` step with a confusing
 * ENOENT — fail at download time instead, with a message that says which
 * pruned directory the symlink was reaching into.
 */
function failOnDanglingSymlinks() {
  const broken = [];
  const stack = [outDir];
  while (stack.length) {
    const next = stack.pop();
    let entries;
    try {
      entries = readdirSync(next, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      const p = join(next, entry.name);
      if (entry.isSymbolicLink()) {
        try {
          // statSync follows the link; if the target is gone this throws.
          statSync(p);
        } catch {
          let target = "<unreadable>";
          try {
            if (lstatSync(p).isSymbolicLink()) target = readlinkSync(p);
          } catch {
            // ignore — best-effort labelling
          }
          broken.push(`${p} → ${target}`);
        }
      } else if (entry.isDirectory()) {
        stack.push(p);
      }
    }
  }
  if (broken.length) {
    console.error(
      "[download-node] Dangling symlinks remain after pruning — these would " +
        "crash electron-builder later with ENOENT. Add the dangling symlink " +
        "(or its target) to pruneUnusedFiles() in this script:",
    );
    for (const entry of broken) console.error("  •", entry);
    throw new Error(`${broken.length} dangling symlink(s) in resources/node/`);
  }
}

// ── Size report ──────────────────────────────────────────────────────────────

function dirSizeBytes(dir) {
  let total = 0;
  const stack = [dir];
  while (stack.length) {
    const next = stack.pop();
    let entries;
    try {
      entries = readdirSync(next, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      const p = join(next, entry.name);
      if (entry.isDirectory()) {
        stack.push(p);
      } else {
        try {
          total += statSync(p).size;
        } catch {}
      }
    }
  }
  return total;
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  const version = resolveVersion();
  const spec = getPlatformSpec(version);
  const archiveName = `${spec.name}.${spec.ext}`;
  const url = `https://nodejs.org/dist/v${version}/${archiveName}`;
  const tmpFile = join(tmpdir(), `node-download-${Date.now()}.${spec.ext}`);

  console.log(
    `[download-node] Downloading Node v${version} for ${PLATFORM}/${ARCH}`,
  );
  console.log(`[download-node] URL: ${url}`);

  try {
    // Clear any previous output so stale files (different Node version, or
    // a stale resources/npm/ from the previous wrapper approach) don't
    // linger in the bundle.
    if (existsSync(outDir)) rmSync(outDir, { recursive: true, force: true });
    mkdirSync(outDir, { recursive: true });

    console.log(`[download-node] Downloading to ${tmpFile}`);
    await downloadFile(url, tmpFile);
    console.log(`[download-node] Extracting to ${outDir}`);
    extract(tmpFile, outDir, spec.ext);

    verifyLayout();
    pruneUnusedFiles();
    failOnDanglingSymlinks();

    const mb = Math.round(dirSizeBytes(outDir) / (1024 * 1024));
    console.log(`[download-node] ✓ Node v${version} ready at ${outDir} (~${mb} MB)`);
  } finally {
    try {
      rmSync(tmpFile, { force: true });
    } catch {}
  }
}

main().catch((err) => {
  console.error("[download-node] Error:", err.message);
  process.exit(1);
});
