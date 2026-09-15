#!/usr/bin/env node
/**
 * Download the doctl binary for the current platform into resources/bin/
 * so that electron-builder can bundle it as an extraResource.
 *
 * The desktop app's MARS port-forward tunnel client (scripts/tunnel-client.mjs)
 * shells out to `doctl agents port-forward` rather than reimplementing the
 * WebSocket client in JS. `agents port-forward` only ships on doctl's
 * published pre-release betas (built off the unmerged feat/agents-subcommands
 * branch) as of 2026-09 — there is no stable release with it yet — so this
 * script resolves against GitHub's full releases list (which includes
 * prereleases), not the /releases/latest endpoint the download-uv.mjs script
 * uses.
 *
 * Usage:
 *   node scripts/download-doctl.mjs                      # latest release, prereleases included
 *   DOCTL_VERSION=v1.168.0-beta.12 node scripts/download-doctl.mjs
 *
 * Output:
 *   resources/bin/doctl     (macOS / Linux)
 *   resources/bin/doctl.exe (Windows)
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
    return {
      os: "darwin",
      arch: ARCH === "arm64" ? "arm64" : "amd64",
      ext: "tar.gz",
      binary: "doctl",
    };
  }
  if (PLATFORM === "linux") {
    return {
      os: "linux",
      arch: ARCH === "arm64" ? "arm64" : "amd64",
      ext: "tar.gz",
      binary: "doctl",
    };
  }
  if (PLATFORM === "win32") {
    return {
      os: "windows",
      arch: "amd64",
      ext: "zip",
      binary: "doctl.exe",
    };
  }
  throw new Error(`Unsupported platform for doctl download: ${PLATFORM}`);
}

// ── Version resolution ────────────────────────────────────────────────────────

async function resolveVersion() {
  if (process.env.DOCTL_VERSION) {
    return process.env.DOCTL_VERSION.replace(/^v/, "");
  }

  console.log(
    "[download-doctl] Fetching latest doctl release (prereleases included) from GitHub API...",
  );
  const headers = { "User-Agent": "agent-canvas-build" };
  // Unauthenticated api.github.com calls are rate-limited per IP (60/hour) —
  // shared CI runner IPs exhaust that fast. CI passes GITHUB_TOKEN.
  if (process.env.GITHUB_TOKEN) {
    headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
  }
  // /releases (not /releases/latest) is required because /releases/latest
  // skips prereleases, and every doctl build with `agents port-forward` is
  // a prerelease beta. The API returns releases newest-first.
  const releases = await fetchJson(
    "https://api.github.com/repos/digitalocean/doctl/releases?per_page=10",
    headers,
  );
  const version = releases[0]?.tag_name?.replace(/^v/, "");
  if (!version) throw new Error("Could not parse doctl version from GitHub API");
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
  // doctl's release archives (both tar.gz and zip) are flat — the binary
  // sits at the archive root, no top-level `doctl-<version>-<target>/`
  // wrapper the way uv's tar.gz does — so no --strip-components is needed.
  // macOS/Linux tar natively supports both; Windows 10+'s built-in bsdtar
  // does too.
  execFileSync("tar", ["-xf", archivePath, "-C", targetDir], { stdio: "inherit" });
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  const spec = getPlatformSpec();
  const version = await resolveVersion();

  console.log(`[download-doctl] Downloading doctl v${version} for ${PLATFORM}/${ARCH}`);

  const filename = `doctl-${version}-${spec.os}-${spec.arch}.${spec.ext}`;
  const url = `https://github.com/digitalocean/doctl/releases/download/v${version}/${filename}`;
  const tmpFile = join(tmpdir(), `doctl-download-${Date.now()}.${spec.ext}`);
  const extractDir = join(tmpdir(), `doctl-extract-${Date.now()}`);

  try {
    mkdirSync(outDir, { recursive: true });
    mkdirSync(extractDir, { recursive: true });

    console.log(`[download-doctl] URL: ${url}`);
    await downloadFile(url, tmpFile);
    console.log(`[download-doctl] Extracting to ${outDir}...`);
    extract(tmpFile, extractDir, spec.ext);

    const src = join(extractDir, spec.binary);
    const dest = join(outDir, spec.binary);

    if (!existsSync(src)) {
      throw new Error(`Expected binary not found after extraction: ${src}`);
    }

    // copyFileSync works across filesystems (unlike renameSync with EXDEV)
    copyFileSync(src, dest);

    if (process.platform !== "win32") {
      chmodSync(dest, 0o755);
    }

    console.log(`[download-doctl] ✓ ${dest}`);
    console.log("[download-doctl] Done. Binary is ready for bundling.");
  } finally {
    // Clean up temp files (best-effort)
    try { rmSync(tmpFile, { force: true }); } catch {}
    try { rmSync(extractDir, { recursive: true, force: true }); } catch {}
  }
}

main().catch((err) => {
  console.error("[download-doctl] Error:", err.message);
  process.exit(1);
});
