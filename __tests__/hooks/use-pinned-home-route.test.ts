import { describe, expect, it } from "vitest";
import {
  clearPinnedExtensionRoute,
  clearStalePinnedExtensionRoutes,
  EXTENSION_PATH_PREFIX,
  getExtensionNameFromPath,
  getPinnedHomeRouteKey,
  isPinnableRoute,
  PINNED_HOME_ROUTE_KEY,
} from "#/hooks/use-pinned-home-route";

describe("getPinnedHomeRouteKey", () => {
  it("scopes the storage key by backend and org", () => {
    expect(getPinnedHomeRouteKey("backend-a", "org-1")).toBe(
      `${PINNED_HOME_ROUTE_KEY}:backend-a:org-1`,
    );
    expect(getPinnedHomeRouteKey("backend-a", null)).toBe(
      `${PINNED_HOME_ROUTE_KEY}:backend-a:-`,
    );
    expect(getPinnedHomeRouteKey("backend-a", "org-1")).not.toBe(
      getPinnedHomeRouteKey("backend-b", "org-1"),
    );
  });
});

describe("isPinnableRoute", () => {
  it("accepts the Customize entry on every backend", () => {
    expect(isPinnableRoute("/customize")).toBe(true);
  });

  it("accepts any Canvas Extension page under the extensions prefix", () => {
    // Single-segment contribution path.
    expect(
      isPinnableRoute(`${EXTENSION_PATH_PREFIX}demo-extension/some-page`),
    ).toBe(true);
    // Nested contribution path is still under the prefix.
    expect(
      isPinnableRoute(`${EXTENSION_PATH_PREFIX}demo-extension/nested/page`),
    ).toBe(true);
  });

  it("rejects the home root so a redirect loop is impossible", () => {
    expect(isPinnableRoute("/")).toBe(false);
  });

  it("rejects arbitrary routes so a stored pin cannot point at a 404", () => {
    // Bare `/extensions` (no extension segment after) is rejected so the
    // pinnable surface is the extension-page set, not the prefix itself.
    expect(isPinnableRoute("/settings")).toBe(false);
    expect(isPinnableRoute("/extensions")).toBe(false);
  });
});

describe("getExtensionNameFromPath", () => {
  it("extracts the decoded extension name from an extension page route", () => {
    expect(
      getExtensionNameFromPath("/extensions/demo-extension/dashboard"),
    ).toBe("demo-extension");
    expect(
      getExtensionNameFromPath("/extensions/my%20extension/sub/page"),
    ).toBe("my extension");
  });

  it("returns null for non-extension routes", () => {
    expect(getExtensionNameFromPath("/customize")).toBeNull();
    expect(getExtensionNameFromPath("/")).toBeNull();
    expect(getExtensionNameFromPath("/settings")).toBeNull();
  });
});

describe("clearPinnedExtensionRoute", () => {
  it("removes the pinned route when it matches the target extension name", () => {
    const key = getPinnedHomeRouteKey("backend-a", "org-1");
    window.localStorage.setItem(
      key,
      JSON.stringify("/extensions/demo-extension/dashboard"),
    );

    clearPinnedExtensionRoute("demo-extension", "backend-a", "org-1");

    expect(window.localStorage.getItem(key)).toBeNull();
  });

  it("leaves the pinned route intact when it targets a different extension or route", () => {
    const key = getPinnedHomeRouteKey("backend-a", "org-1");
    window.localStorage.setItem(
      key,
      JSON.stringify("/extensions/other-extension/dashboard"),
    );

    clearPinnedExtensionRoute("demo-extension", "backend-a", "org-1");

    expect(window.localStorage.getItem(key)).toBe(
      JSON.stringify("/extensions/other-extension/dashboard"),
    );
  });
});

describe("clearStalePinnedExtensionRoutes", () => {
  it("removes a pinned extension route when its extension is disabled/missing", () => {
    const key = getPinnedHomeRouteKey("backend-a", "org-1");
    window.localStorage.setItem(
      key,
      JSON.stringify("/extensions/disabled-ext/page"),
    );

    clearStalePinnedExtensionRoutes("backend-a", "org-1", ["enabled-ext"]);

    expect(window.localStorage.getItem(key)).toBeNull();
  });

  it("preserves the pinned extension route when its extension is enabled", () => {
    const key = getPinnedHomeRouteKey("backend-a", "org-1");
    window.localStorage.setItem(
      key,
      JSON.stringify("/extensions/enabled-ext/page"),
    );

    clearStalePinnedExtensionRoutes("backend-a", "org-1", ["enabled-ext"]);

    expect(window.localStorage.getItem(key)).toBe(
      JSON.stringify("/extensions/enabled-ext/page"),
    );
  });

  it("preserves non-extension pinned routes like /customize", () => {
    const key = getPinnedHomeRouteKey("backend-a", "org-1");
    window.localStorage.setItem(key, JSON.stringify("/customize"));

    clearStalePinnedExtensionRoutes("backend-a", "org-1", []);

    expect(window.localStorage.getItem(key)).toBe(JSON.stringify("/customize"));
  });
});
