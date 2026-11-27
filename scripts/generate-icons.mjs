/**
 * Generate electron/build-resources/icon.ico and icon.icns from the
 * 1024×1024 icon.png master. Run after changing icon.png:
 *
 *   npm run generate-icons
 *
 * The outputs are COMMITTED — electron-builder auto-discovers them in
 * directories.buildResources and uses them as-is. We don't rely on
 * electron-builder's own PNG→ICO conversion because it emits a single
 * 256×256 PNG-compressed entry, which several Windows shell surfaces
 * (Explorer small views, NSIS, taskbar) can't render — they fall back to
 * the default Electron icon. png2icons with forWinExe=true stores the
 * 48/32/24/16 entries as classic BMP, which is what Windows expects.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import png2icons from "png2icons";

// Standard Windows app-icon sizes (Microsoft recommends 16/24/32/48/256;
// 64/128 cover intermediate DPI scaling).
export const REQUIRED_ICO_SIZES = [16, 24, 32, 48, 64, 128, 256];

// The ten standard macOS representations (16 → 512@2x). is32/il32 (+ the
// l8mk/s8mk masks png2icons emits alongside) are the legacy non-retina
// 16/32 forms; ic07–ic14 cover 128 → 512@2x.
export const REQUIRED_ICNS_TYPES = [
  "is32",
  "il32",
  "ic07",
  "ic08",
  "ic09",
  "ic10",
  "ic11",
  "ic12",
  "ic13",
  "ic14",
];

/**
 * Parse an ICO buffer's ICONDIR into [{size, isPng}] — isPng distinguishes
 * PNG-compressed payloads from classic BMP entries.
 */
export function listIcoEntries(ico) {
  const entries = [];
  for (let i = 0; i < ico.readUInt16LE(4); i++) {
    const entry = 6 + i * 16;
    const width = ico[entry];
    const offset = ico.readUInt32LE(entry + 12);
    entries.push({
      size: width === 0 ? 256 : width,
      isPng: ico.readUInt32BE(offset) === 0x89504e47, // \x89PNG
    });
  }
  return entries;
}

/** List the 4-char OSType of every chunk in an ICNS buffer. */
export function listIcnsTypes(icns) {
  const types = [];
  for (let p = 8; p < icns.length; p += icns.readUInt32BE(p + 4)) {
    types.push(icns.toString("ascii", p, p + 4));
  }
  return types;
}

/**
 * Convert a PNG buffer into { ico, icns } buffers, asserting the emitted
 * size sets so a png2icons upgrade that changes them fails loudly.
 */
export function generateIcons(input) {
  // forWinExe: true → 48/32/24/16 stored as classic BMP entries (required
  // by Windows shell small-icon surfaces), ≥64 PNG-compressed. 0 = lossless.
  const ico = png2icons.createICO(input, png2icons.BICUBIC2, 0, false, true);
  const icns = png2icons.createICNS(input, png2icons.BICUBIC2, 0);
  if (!ico || !icns) throw new Error("png2icons failed to convert icon.png");

  const icoSizes = listIcoEntries(ico).map((entry) => entry.size);
  for (const size of REQUIRED_ICO_SIZES) {
    if (!icoSizes.includes(size)) throw new Error(`icon.ico missing ${size}px`);
  }
  const icnsTypes = listIcnsTypes(icns);
  for (const type of REQUIRED_ICNS_TYPES) {
    if (!icnsTypes.includes(type)) throw new Error(`icon.icns missing ${type}`);
  }

  return { ico, icns };
}

const isMainModule =
  process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;

if (isMainModule) {
  const resDir = join(
    dirname(fileURLToPath(import.meta.url)),
    "..",
    "electron",
    "build-resources",
  );
  const { ico, icns } = generateIcons(readFileSync(join(resDir, "icon.png")));
  writeFileSync(join(resDir, "icon.ico"), ico);
  writeFileSync(join(resDir, "icon.icns"), icns);
  console.log(
    `icon.ico (${listIcoEntries(ico)
      .map((entry) => entry.size)
      .join("/")}px, ${ico.length} B), ` +
      `icon.icns (${listIcnsTypes(icns).join(",")}, ${icns.length} B)`,
  );
}
