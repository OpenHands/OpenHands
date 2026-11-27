// @vitest-environment node
// The desktop app icons are COMMITTED artifacts (electron/build-resources/
// icon.ico + icon.icns) that electron-builder auto-discovers and uses as-is.
// We ship them instead of relying on electron-builder's PNG→ICO conversion
// because that conversion emits a single 256×256 PNG-compressed entry which
// several Windows shell surfaces can't render — the app then shows the
// default Electron icon. These tests guard the structural contract of the
// committed files (standard sizes, BMP small entries), that they stay in
// sync with the icon.png master, and that the builder config keeps shipping
// the runtime window icons into the packaged app.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import config from "../../electron-builder.config.mjs";
import {
  REQUIRED_ICNS_TYPES,
  REQUIRED_ICO_SIZES,
  generateIcons,
  listIcnsTypes,
  listIcoEntries,
} from "../../scripts/generate-icons.mjs";

const resDir = fileURLToPath(
  new URL("../../electron/build-resources", import.meta.url),
);
const committedPng = readFileSync(join(resDir, "icon.png"));
const committedIco = readFileSync(join(resDir, "icon.ico"));
const committedIcns = readFileSync(join(resDir, "icon.icns"));

describe("committed desktop icon artifacts", () => {
  it("icon.ico contains every standard Windows size", () => {
    const sizes = listIcoEntries(committedIco).map((entry) => entry.size);
    for (const size of REQUIRED_ICO_SIZES) {
      expect(sizes, `missing ${size}px entry`).toContain(size);
    }
  });

  it("icon.ico stores 48px-and-below entries as classic BMP, not PNG", () => {
    // The root cause of the Windows default-icon bug: small entries stored
    // as PNG-compressed data, which Explorer/NSIS/taskbar can't render.
    const smallEntries = listIcoEntries(committedIco).filter(
      (entry) => entry.size <= 48,
    );
    expect(smallEntries.length).toBeGreaterThan(0);
    for (const entry of smallEntries) {
      expect(entry.isPng, `${entry.size}px entry must be BMP`).toBe(false);
    }
  });

  it("icon.icns contains all ten standard macOS representations", () => {
    const types = listIcnsTypes(committedIcns);
    for (const type of REQUIRED_ICNS_TYPES) {
      expect(types, `missing ${type} representation`).toContain(type);
    }
  });

  it("stays in sync with the icon.png master (npm run generate-icons)", () => {
    // Fails when icon.png changes without regenerating the committed
    // artifacts — generation is deterministic, so bytes must match exactly.
    const { ico, icns } = generateIcons(committedPng);
    expect(ico.equals(committedIco)).toBe(true);
    expect(icns.equals(committedIcns)).toBe(true);
  });
});

describe("electron-builder icon wiring", () => {
  it("points buildResources at the directory holding the committed icons", () => {
    // Auto-discovery contract: icon.icns (mac), icon.ico (win) and icon.png
    // (linux) are all picked up from directories.buildResources by name.
    expect(config.directories?.buildResources).toBe(
      "electron/build-resources",
    );
  });

  it("ships both runtime window icons into the packaged app", () => {
    // buildResources is auto-excluded from packaging, so main.mjs's
    // BrowserWindow icons (icon.ico on win32, icon.png elsewhere) need
    // explicit re-includes to exist on disk at runtime.
    expect(config.files).toContain("build-resources/icon.png");
    expect(config.files).toContain("build-resources/icon.ico");
  });
});
