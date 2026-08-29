import { describe, expect, it } from "vitest";
import {
  EXTENSION_PATH_PREFIX,
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
