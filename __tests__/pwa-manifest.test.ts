import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { matchRoutes, type RouteObject } from "react-router";
import { describe, expect, it } from "vitest";
import routes from "../src/routes";

const manifest = JSON.parse(
  readFileSync(resolve("public/site.webmanifest"), "utf8"),
);

function routeObjects(entries: typeof routes): RouteObject[] {
  return entries.map((entry) =>
    entry.index
      ? { index: true }
      : {
          path: entry.path,
          children: entry.children ? routeObjects(entry.children) : undefined,
        },
  );
}

// @spec AWI-002 — Installed-app launches and shortcuts
describe("installed-app manifest", () => {
  it("requests navigation of the existing app window", () => {
    expect(manifest.launch_handler?.client_mode).toBe("navigate-existing");
    expect(manifest.handle_links).toBe("preferred");
  });

  it.each(["/", "/canvas/"])(
    "resolves shortcuts and icons to Canvas resources mounted at %s",
    (basePath) => {
      const manifestUrl = new URL(
        `${basePath}site.webmanifest`,
        "https://canvas.example",
      );
      const shortcuts = manifest.shortcuts as {
        name: string;
        url: string;
        icons: { src: string }[];
      }[];
      expect(shortcuts).toHaveLength(3);

      for (const shortcut of shortcuts) {
        expect(shortcut.name.trim()).not.toBe("");
        expect(shortcut.url).not.toMatch(/^(?:\/|[a-z][a-z\d+.-]*:)/i);
        const target = new URL(shortcut.url, manifestUrl);
        expect(target.origin).toBe(manifestUrl.origin);
        expect(target.pathname.startsWith(basePath)).toBe(true);
        const appPath = `/${target.pathname.slice(basePath.length)}`;
        const matches = matchRoutes(routeObjects(routes), appPath);
        expect(matches).not.toBeNull();
        expect(matches?.at(-1)?.route.path).not.toBe("*");

        expect(shortcut.icons.length).toBeGreaterThan(0);
        for (const icon of shortcut.icons) {
          const iconUrl = new URL(icon.src, manifestUrl);
          expect(iconUrl.pathname.startsWith(basePath)).toBe(true);
          expect(
            existsSync(
              resolve("public", iconUrl.pathname.slice(basePath.length)),
            ),
          ).toBe(true);
        }
      }
    },
  );
});
