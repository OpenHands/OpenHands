// @vitest-environment node
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  DEFAULT_COLOR_THEME,
  getColorThemeBaseColor,
} from "#/themes/color-themes";

const PUBLIC_DIR = resolve(__dirname, "../public");

interface ManifestIcon {
  src: string;
  sizes: string;
  type: string;
  purpose?: string;
}

const manifest = JSON.parse(
  readFileSync(resolve(PUBLIC_DIR, "site.webmanifest"), "utf8"),
) as {
  name: string;
  short_name: string;
  description: string;
  start_url: string;
  scope: string;
  display: string;
  theme_color: string;
  background_color: string;
  categories: string[];
  icons: ManifestIcon[];
};

/** Width/height out of a PNG's IHDR chunk, which always leads the file. */
function readPngSize(path: string): { width: number; height: number } {
  const header = readFileSync(path).subarray(0, 24);
  return {
    width: header.readUInt32BE(16),
    height: header.readUInt32BE(20),
  };
}

describe("web app manifest", () => {
  it("declares the fields a browser needs to offer installation", () => {
    expect(manifest.name).toBe("OpenHands Agent Canvas");
    expect(manifest.short_name).toBe("Agent Canvas");
    expect(manifest.short_name.length).toBeLessThanOrEqual(12);
    expect(manifest.description).not.toBe("");
    expect(manifest.display).toBe("standalone");
    expect(manifest.categories.length).toBeGreaterThan(0);
  });

  // Canvas can be mounted under a subpath (`agent-canvas --base-path /canvas`),
  // where the manifest is fetched from `/canvas/site.webmanifest`. Members are
  // resolved against the manifest's own URL, so keeping every one of them
  // relative is what makes a mounted install work — a leading `/` would send
  // `start_url` and the icons back to the origin root.
  describe("URL members stay relative to the manifest", () => {
    const urlMembers = [
      ["start_url", manifest.start_url],
      ["scope", manifest.scope],
      ...manifest.icons.map((icon): [string, string] => [
        `icons[${icon.src}]`,
        icon.src,
      ]),
    ] as const;

    it.each(urlMembers)("%s is relative", (_name, value) => {
      expect(value.startsWith("/")).toBe(false);
      expect(/^[a-z][a-z\d+.-]*:/i.test(value)).toBe(false);
    });

    it("resolves under the mount point when Canvas is served from a subpath", () => {
      const manifestUrl = new URL(
        "https://example.test/canvas/site.webmanifest",
      );

      expect(new URL(manifest.start_url, manifestUrl).pathname).toBe(
        "/canvas/",
      );
      expect(new URL(manifest.scope, manifestUrl).pathname).toBe("/canvas/");
      for (const icon of manifest.icons) {
        expect(new URL(icon.src, manifestUrl).pathname).toBe(
          `/canvas/${icon.src}`,
        );
      }
    });
  });

  describe("icons", () => {
    it("ships every referenced icon at its declared size", () => {
      for (const icon of manifest.icons) {
        const path = resolve(PUBLIC_DIR, icon.src);
        expect(existsSync(path), `${icon.src} is missing`).toBe(true);

        if (icon.type !== "image/png") continue;
        const [width, height] = icon.sizes.split("x").map(Number);
        expect(readPngSize(path), `${icon.src} size`).toEqual({
          width,
          height,
        });
      }
    });

    it("covers the sizes Chromium requires for installability", () => {
      const anySizes = manifest.icons
        .filter((icon) => icon.purpose !== "maskable")
        .map((icon) => icon.sizes);

      expect(anySizes).toContain("192x192");
      expect(anySizes).toContain("512x512");
    });

    // Without one, Android crops the square icon to fit its adaptive mask and
    // clips the artwork instead of the padding.
    it("ships a maskable icon", () => {
      const maskable = manifest.icons.filter(
        (icon) => icon.purpose === "maskable",
      );

      expect(maskable).toHaveLength(1);
      expect(maskable[0].sizes).toBe("512x512");
    });

    // iOS ignores the manifest icons and reads this one, at this exact size.
    it("ships a 180x180 apple-touch-icon", () => {
      expect(readPngSize(resolve(PUBLIC_DIR, "apple-touch-icon.png"))).toEqual({
        width: 180,
        height: 180,
      });
    });
  });

  // The splash screen and title bar are painted from the manifest before any
  // app code runs, so these have to be the page background the default theme
  // paints — otherwise launching flashes a different color than it settles on.
  it("paints the title bar and splash in the default theme's base color", () => {
    const baseColor = getColorThemeBaseColor(DEFAULT_COLOR_THEME);

    expect(manifest.theme_color.toLowerCase()).toBe(baseColor.toLowerCase());
    expect(manifest.background_color.toLowerCase()).toBe(
      baseColor.toLowerCase(),
    );
  });
});
