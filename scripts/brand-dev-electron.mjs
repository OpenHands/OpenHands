#!/usr/bin/env node
/**
 * Brand the dev Electron app bundle with the product name (macOS only).
 *
 * Run automatically as the `predesktop` npm hook.
 *
 * WHY THIS EXISTS
 *
 * `npm run desktop` runs the app inside Electron's own prebuilt bundle,
 * node_modules/electron/dist/Electron.app. macOS derives the name it shows
 * in the Dock tooltip, the ⌘-Tab switcher and Finder from that bundle, at
 * launch, before any JavaScript runs. No runtime API can change it —
 * `app.setName()`, `app.name` and package.json `productName` only drive
 * Electron's own notion of the name (menu bar, About panel, and
 * app.getPath("userData")).
 *
 * The packaged app has never had this problem: electron-builder emits a
 * bundle literally named "<productName>.app" with matching plist keys. This
 * script puts the dev bundle in that same state.
 *
 * THE BUNDLE FILENAME IS THE PART THAT ACTUALLY SHOWS
 *
 * Patching the plist alone is NOT enough — verified the hard way. macOS
 * prefers the bundle's filesystem name over CFBundleName/CFBundleDisplayName
 * for the Dock tooltip. Two installed apps prove each half of this:
 *
 *   DBeaver.app        CFBundleName "DBeaver Community" → displays "DBeaver"
 *                      (the filename wins over the plist)
 *   Antigravity.app    CFBundleExecutable "Electron"    → displays "Antigravity"
 *                      (an Electron app whose executable name is irrelevant)
 *
 * So the fix aligns every source of the name at once: the .app directory
 * name, CFBundleName and CFBundleDisplayName. That is exactly the shape of a
 * packaged build, which is known to display correctly.
 *
 * Renaming the bundle means node_modules/electron/path.txt has to move with
 * it: getElectronPath() in node_modules/electron/index.js joins path.txt onto
 * dist/ and silently re-downloads Electron (~100 MB) if the result does not
 * exist. The two are updated together, and the rename is rolled back if
 * path.txt cannot be written.
 *
 * WHY THIS IS SAFE
 *
 *   - Electron's prebuilt dist is ad-hoc *linker-signed*: `codesign -dv`
 *     reports `flags=0x20002(adhoc,linker-signed)`, `Info.plist=not bound`,
 *     `Sealed Resources=none`. The signature covers only the Mach-O, so
 *     editing Info.plist does not invalidate it and no re-signing is needed.
 *   - `npm run build:desktop` is unaffected: electron-builder packages from
 *     its own download cache (~/Library/Caches/electron/electron-v*.zip),
 *     never from node_modules/electron/dist.
 *
 * WHAT IT DELIBERATELY DOES NOT TOUCH
 *
 *   CFBundleExecutable — left as "Electron". Antigravity above shows it has
 *     no bearing on the displayed name; it only feeds ps / Activity Monitor.
 *     Leaving it alone keeps path.txt's trailing segments valid.
 *   CFBundleIdentifier — kept at com.github.Electron. Changing it would split
 *     LaunchServices / TCC state per checkout for no visible gain.
 *
 * The changes live in node_modules, which `npm ci` wipes. That is fine: the
 * `predesktop` hook re-applies them on every `npm run desktop`.
 *
 * This script must never block the desktop run — every failure path warns
 * and exits 0.
 */

import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, "..");

// Both keys are set. CFBundleDisplayName is the one LaunchServices reports;
// CFBundleName is the fallback and shows up in other bundle-name surfaces.
const NAME_KEYS = ["CFBundleDisplayName", "CFBundleName"];

function warn(message) {
  console.warn(`[brand-dev-electron] ${message}`);
}

/** Root of the installed `electron` package, or null if it isn't resolvable. */
function resolveElectronPackage() {
  const require = createRequire(import.meta.url);
  return dirname(require.resolve("electron/package.json"));
}

/** Current value of `key`, or null when the key is absent. */
function readPlistString(plistPath, key) {
  try {
    return execFileSync(
      "plutil",
      ["-extract", key, "raw", "-o", "-", plistPath],
      { encoding: "utf8" },
    ).trim();
  } catch {
    return null;
  }
}

function writePlistString(plistPath, key, value) {
  execFileSync("plutil", ["-replace", key, "-string", value, plistPath], {
    stdio: "pipe",
  });
}

/**
 * Rename dist/<current>.app to dist/<productName>.app and repoint path.txt.
 * Returns { appDir, renamed }, or null if the bundle can't be determined.
 */
function ensureBundleName(pkgDir, productName) {
  const distDir = join(pkgDir, "dist");
  const pathFile = join(pkgDir, "path.txt");
  if (!existsSync(pathFile)) {
    warn(`No ${pathFile} — leaving the dev bundle alone.`);
    return null;
  }

  // e.g. "Electron.app/Contents/MacOS/Electron" — only the first segment
  // (the bundle directory) is ours to rename.
  const relative = readFileSync(pathFile, "utf8").trim();
  const segments = relative.split("/");
  const currentName = segments[0];
  const desiredName = `${productName}.app`;
  if (!currentName.endsWith(".app")) {
    warn(`Unexpected path.txt entry "${relative}" — leaving the bundle alone.`);
    return null;
  }

  const desiredDir = join(distDir, desiredName);
  if (currentName === desiredName && existsSync(desiredDir)) {
    return { appDir: desiredDir, renamed: false };
  }

  const currentDir = join(distDir, currentName);
  let renamed = false;
  if (existsSync(currentDir) && currentDir !== desiredDir) {
    if (existsSync(desiredDir)) {
      // Both present: a previous run renamed the bundle and something
      // restored the original. Prefer the correctly named one and just fix
      // path.txt rather than clobbering either bundle.
      warn(`Both ${currentName} and ${desiredName} exist — using the latter.`);
    } else {
      renameSync(currentDir, desiredDir);
      renamed = true;
    }
  } else if (!existsSync(desiredDir)) {
    warn(`No Electron bundle under ${distDir} — leaving the dev bundle alone.`);
    return null;
  }

  // path.txt MUST agree with the directory on disk; a stale entry makes
  // getElectronPath() re-download Electron on the next run.
  segments[0] = desiredName;
  try {
    writeFileSync(pathFile, segments.join("/"));
  } catch (err) {
    // Undo the rename so the checkout is left in a working state.
    if (existsSync(desiredDir) && !existsSync(currentDir)) {
      try {
        renameSync(desiredDir, currentDir);
      } catch {
        warn(`Could not roll back the bundle rename in ${distDir}.`);
      }
    }
    warn(`Could not update ${pathFile}: ${err.message}`);
    return null;
  }

  return { appDir: desiredDir, renamed };
}

function main() {
  // The Dock/⌘-Tab name is a macOS bundle concept. On Windows the dev
  // taskbar name comes from electron.exe's version resource and on Linux
  // from the .desktop file / WM_CLASS — neither exists until the app is
  // packaged, so there is nothing to patch.
  if (process.platform !== "darwin") return;

  const manifestPath = join(projectRoot, "electron", "package.json");
  let productName;
  try {
    productName = JSON.parse(readFileSync(manifestPath, "utf8")).productName;
  } catch (err) {
    warn(`Could not read ${manifestPath}: ${err.message}`);
    return;
  }
  if (!productName) {
    warn(`No productName in ${manifestPath} — leaving the dev bundle alone.`);
    return;
  }
  // The name becomes a directory entry; a "/" would silently retarget it.
  if (productName.includes("/")) {
    warn(`productName "${productName}" cannot be used as a bundle name.`);
    return;
  }

  let pkgDir;
  try {
    pkgDir = resolveElectronPackage();
  } catch (err) {
    warn(`Could not resolve the electron package: ${err.message}`);
    return;
  }

  let bundle;
  try {
    bundle = ensureBundleName(pkgDir, productName);
  } catch (err) {
    warn(`Could not rename the dev bundle: ${err.message}`);
    return;
  }
  if (!bundle) return;

  const plistPath = join(bundle.appDir, "Contents", "Info.plist");
  if (!existsSync(plistPath)) {
    warn(`No Info.plist at ${plistPath} — leaving the plist alone.`);
    return;
  }

  const stale = NAME_KEYS.filter(
    (key) => readPlistString(plistPath, key) !== productName,
  );

  try {
    for (const key of stale) writePlistString(plistPath, key, productName);
  } catch (err) {
    // Read-only node_modules (CI caches, sandboxes) lands here. The app still
    // runs; only the displayed name keeps saying "Electron".
    warn(`Could not patch ${plistPath}: ${err.message}`);
    return;
  }

  // Stay quiet when there was nothing to do, so repeat runs don't add noise.
  if (bundle.renamed || stale.length > 0) {
    console.log(
      `[brand-dev-electron] Dev Electron bundle now identifies as "${productName}".`,
    );
  }
}

main();
