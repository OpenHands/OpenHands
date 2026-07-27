#!/usr/bin/env node
/**
 * Download the uv binary for the current platform into resources/bin/
 * so that electron-builder can bundle it as an extraResource.
 *
 * uv provides uvx, which the Electron desktop app uses to run the
 * agent-server and automation backend Python packages.
 *
 * Usage:
 *   node scripts/download-uv.mjs          # uses latest GitHub release
 *   UV_VERSION=0.7.0 node scripts/download-uv.mjs
 *
 * Output (per platform):
 *   resources/bin/uv    + resources/bin/uvx     (macOS / Linux)
 *   resources/bin/uv.exe + resources/bin/uvx.exe (Windows)
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
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, "..");
const outDir = join(projectRoot, "resources", "bin");

// ── Platform detection ─────────────────────────────────────────────────────────

const PLATFORM = process.platform; // 'darwin' | 'linux' | 'win32'
const ARCH = process.arch;         // 'x64' | 'arm64'

function getPlatformSpec() {
  if (PLATFORM === "darwin") {
    const uvArch = ARCH === "arm64" ? "aarch64" : "x86_64";
    return {
      target: `uv-${uvArch}-apple-darwin`,
      ext: "tar.gz",
      binaries: ["uv", "uvx"],
    };
  }
  if (PLATFORM === "linux") {
    // Only x64 is officially supported by uv for desktop builds
    return {
      target: "uv-x86_64-unknown-linux-gnu",
      ext: "tar.gz",
      binaries: ["uv", "uvx"],
    };
  }
  if (PLATFORM === "win32") {
    return {
      target: "uv-x86_64-pc-windows-msvc",
      ext: "zip",
      binaries: ["uv.exe", "uvx.exe"],
    };
  }
  throw new Error(`Unsupported platform for uv download: ${PLATFORM}`);
}

// ── Version resolution ────────────────────────────────────────────────────────

async function resolveVersion() {
  if (process.env.UV_VERSION) {
    return process.env.UV_VERSION.replace(/^v/, "");
  }

  console.log("[download-uv] Fetching latest uv version from GitHub API...");
  const headers = { "User-Agent": "agent-canvas-build" };
  // Unauthenticated api.github.com calls are rate-limited per IP (60/hour) —
  // shared CI runner IPs exhaust that fast. CI passes GITHUB_TOKEN.
  if (process.env.GITHUB_TOKEN) {
    headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
  }
  const data = await fetchJson(
    "https://api.github.com/repos/astral-sh/uv/releases/latest",
    headers
  );
  const version = data.tag_name?.replace(/^v/, "");
  if (!version) throw new Error("Could not parse uv version from GitHub API");
  return version;
}

// ── HTTP helpers ──────────────────────────────────────────────────────────────

function fetchJson(url, headers = {}) {
  return new Promise((resolve, reject) => {
    get(url, { headers }, (res) => {
      if (res.statusCode === 301 || res.statusCode === 302) {
        return resolve(fetchJson(res.headers.location, headers));
      }
      if (res.statusCode !== 200) {
        return reject(new Error(`GET ${url} → HTTP ${res.statusCode}`));
      }
      let body = "";
      res.on("data", (chunk) => (body += chunk));
      res.on("end", () => resolve(JSON.parse(body)));
      res.on("error", reject);
    }).on("error", reject);
  });
}

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
  const spec = getPlatformSpec();
  const version = await resolveVersion();

  console.log(`[download-uv] Downloading uv v${version} for ${PLATFORM}/${ARCH}`);

  const filename = `${spec.target}.${spec.ext}`;
  const url = `https://github.com/astral-sh/uv/releases/download/${version}/${filename}`;
  const tmpFile = join(tmpdir(), `uv-download-${Date.now()}.${spec.ext}`);
  const extractDir = join(tmpdir(), `uv-extract-${Date.now()}`);

  try {
    mkdirSync(outDir, { recursive: true });
    mkdirSync(extractDir, { recursive: true });

    console.log(`[download-uv] URL: ${url}`);
    await downloadFile(url, tmpFile);
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

    console.log("[download-uv] Done. Binaries are ready for bundling.");
  } finally {
    // Clean up temp files (best-effort)
    try { rmSync(tmpFile, { force: true }); } catch {}
    try { rmSync(extractDir, { recursive: true, force: true }); } catch {}
  }
}

main().catch((err) => {
  console.error("[download-uv] Error:", err.message);
  process.exit(1);
});
