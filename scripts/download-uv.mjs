#!/usr/bin/env node
/**
 * Download the uv binary for the current platform into resources/bin/
 * so that electron-builder can bundle it as an extraResource.
 *
 * uv provides uvx, which the Electron desktop app uses to run the
 * agent-server and automation backend Python packages.
 *
 * Usage:
 *   node scripts/download-uv.mjs          # uses the pinned UV_BUNDLE_VERSION
 *   UV_VERSION=0.7.0 node scripts/download-uv.mjs
 *   ELECTRON_ARCH=universal node scripts/download-uv.mjs   # macOS only
 *
 * Every download is verified against the release's published .sha256 asset
 * before extraction (see scripts/download-integrity.mjs) — a mismatch or a
 * missing checksum entry aborts the build, including UV_VERSION overrides.
 *
 * Output (per platform):
 *   resources/bin/uv    + resources/bin/uvx     (macOS / Linux)
 *   resources/bin/uv.exe + resources/bin/uvx.exe (Windows)
 *
 * With ELECTRON_ARCH=universal on macOS, both darwin slices are downloaded
 * into per-arch directories — the multi-arch layout is canonical in
 * scripts/download-arch-utils.mjs.
 */

import {
  chmodSync,
  copyFileSync,
  createWriteStream,
  existsSync,
  mkdirSync,
  rmSync,
} from "node:fs";
import { get } from "node:https";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { execFileSync } from "node:child_process";
import { resolveDownloadArches, resourceDirName } from "./download-arch-utils.mjs";
import { verifyChecksum } from "./download-integrity.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, "..");

// ── Platform detection ─────────────────────────────────────────────────────────

const PLATFORM = process.platform; // 'darwin' | 'linux' | 'win32'
const ARCH = process.arch;         // 'x64' | 'arm64'

/**
 * Map arch → uv's published release target for the given platform.
 * `arch` is a Node-style arch ("arm64" | "x64"); uv's Rust-style triplets
 * use "aarch64" / "x86_64".
 */
export function getPlatformSpec(arch = ARCH, platform = PLATFORM) {
  if (platform === "darwin") {
    const uvArch = arch === "arm64" ? "aarch64" : "x86_64";
    return {
      target: `uv-${uvArch}-apple-darwin`,
      ext: "tar.gz",
      binaries: ["uv", "uvx"],
    };
  }
  if (platform === "linux") {
    // Only x64 is officially supported by uv for desktop builds
    return {
      target: "uv-x86_64-unknown-linux-gnu",
      ext: "tar.gz",
      binaries: ["uv", "uvx"],
    };
  }
  if (platform === "win32") {
    return {
      target: "uv-x86_64-pc-windows-msvc",
      ext: "zip",
      binaries: ["uv.exe", "uvx.exe"],
    };
  }
  throw new Error(`Unsupported platform for uv download: ${platform}`);
}

// ── Version resolution ────────────────────────────────────────────────────────

// Keep the bundled uv pinned for reproducible, tamper-evident builds — same
// convention as NODE_BUNDLE_VERSION in download-node.mjs. The pin follows
// the repository's dependency release-age policy (0.13.8 was >= 7 days old
// when pinned, 2026-11-22). Override at build time with UV_VERSION=… to
// test another release; the override is checksum-verified like the default.
const UV_BUNDLE_VERSION = "0.13.8";

function resolveVersion() {
  const requested = process.env.UV_VERSION?.replace(/^v/, "");
  return requested || UV_BUNDLE_VERSION;
}

// ── HTTP helpers ──────────────────────────────────────────────────────────────

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

/** GET a small text resource (the release's .sha256 asset), following redirects. */
function downloadText(url) {
  return new Promise((resolve, reject) => {
    function doGet(u) {
      get(u, { headers: { "User-Agent": "agent-canvas-build" } }, (res) => {
        if (res.statusCode === 301 || res.statusCode === 302) {
          return doGet(res.headers.location);
        }
        if (res.statusCode !== 200) {
          return reject(new Error(`GET ${u} → HTTP ${res.statusCode}`));
        }
        let body = "";
        res.on("data", (chunk) => (body += chunk));
        res.on("end", () => resolve(body));
        res.on("error", reject);
      }).on("error", reject);
    }
    doGet(url);
  });
}

// ── Extraction ────────────────────────────────────────────────────────────────

function extract(archivePath, targetDir, ext) {
  // Both tar.gz and zip are handled by the system 'tar' command:
  //   macOS/Linux: GNU/BSD tar natively supports .tar.gz
  //   Windows 10+: built-in bsdtar supports both .tar.gz and .zip
  // The tar.gz archives wrap uv/uvx in a top-level `uv-<target>/` directory,
  // which --strip-components=1 removes. uv's Windows .zip is flat (uv.exe /
  // uvx.exe at the archive root) — stripping there would skip every entry
  // and extract nothing.
  const args = ["-xf", archivePath, "-C", targetDir];
  if (ext === "tar.gz") {
    args.push("--strip-components=1");
  }
  execFileSync("tar", args, { stdio: "inherit" });
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  // Resolve the version once (one UV_BUNDLE_VERSION / UV_VERSION read) so
  // every arch in a universal build downloads the exact same release.
  const version = resolveVersion();
  const arches = resolveDownloadArches();
  const multi = arches.length > 1;
  if (multi) {
    console.log(
      `[download-uv] macOS universal build: downloading per-arch uv runtimes for ${arches.join(", ")}`,
    );
  }

  for (const arch of arches) {
    const spec = getPlatformSpec(arch);
    const outDir = join(
      projectRoot,
      "resources",
      resourceDirName("bin", arch, multi),
    );
    const filename = `${spec.target}.${spec.ext}`;
    const url = `https://github.com/astral-sh/uv/releases/download/${version}/${filename}`;
    const tmpFile = join(tmpdir(), `uv-download-${arch}-${Date.now()}.${spec.ext}`);
    const extractDir = join(tmpdir(), `uv-extract-${arch}-${Date.now()}`);

    console.log(`[download-uv] Downloading uv v${version} for ${PLATFORM}/${arch}`);

    try {
      mkdirSync(outDir, { recursive: true });
      mkdirSync(extractDir, { recursive: true });

      console.log(`[download-uv] URL: ${url}`);
      await downloadFile(url, tmpFile);

      // Verify against the release's published .sha256 asset before touching
      // the archive — a mismatched or missing entry aborts the build.
      const checksumUrl = `${url}.sha256`;
      console.log(`[download-uv] Verifying against ${checksumUrl}`);
      verifyChecksum(
        tmpFile,
        filename,
        await downloadText(checksumUrl),
        `uv release asset ${checksumUrl}`,
      );
      console.log(`[download-uv] ✓ checksum verified`);

      console.log(`[download-uv] Extracting to ${outDir}...`);
      extract(tmpFile, extractDir, spec.ext);

      // Copy the required binaries to outDir
      for (const bin of spec.binaries) {
        const src = join(extractDir, bin);
        const dest = join(outDir, bin);

        if (!existsSync(src)) {
          throw new Error(`Expected binary not found after extraction: ${src}`);
        }

        // copyFileSync works across filesystems (unlike renameSync with EXDEV)
        copyFileSync(src, dest);

        if (process.platform !== "win32") {
          chmodSync(dest, 0o755);
        }

        console.log(`[download-uv] ✓ ${dest}`);
      }
    } finally {
      // Clean up temp files (best-effort)
      try { rmSync(tmpFile, { force: true }); } catch {}
      try { rmSync(extractDir, { recursive: true, force: true }); } catch {}
    }
  }

  console.log("[download-uv] Done. Binaries are ready for bundling.");
}

// Run only when executed directly (`node scripts/download-uv.mjs`), not
// when imported — __tests__/scripts/download-uv.test.ts imports this module
// to exercise getPlatformSpec with zero network I/O. process.argv[1] is
// undefined when the module is plain-imported (vitest), and
// pathToFileURL(undefined) would throw, so guard on it first.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err) => {
    console.error("[download-uv] Error:", err.message);
    process.exit(1);
  });
}
