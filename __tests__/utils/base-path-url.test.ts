import { describe, expect, it } from "vitest";
import { withBasePathSlash } from "#/utils/base-path-url";

const at = (pathname: string, search = "", hash = "") => ({
  pathname,
  search,
  hash,
});

describe("withBasePathSlash", () => {
  it("adds the trailing slash to the bare base path", () => {
    expect(withBasePathSlash(at("/canvas"), "/canvas/")).toBe("/canvas/");
  });

  it("keeps the query and hash, as after sign-in", () => {
    expect(
      withBasePathSlash(
        at("/canvas", "?login_method=keycloak", "#top"),
        "/canvas/",
      ),
    ).toBe("/canvas/?login_method=keycloak#top");
  });

  it("leaves URLs the router already matches alone", () => {
    expect(withBasePathSlash(at("/canvas/"), "/canvas/")).toBeNull();
    expect(
      withBasePathSlash(at("/canvas/conversations/abc"), "/canvas/"),
    ).toBeNull();
    expect(withBasePathSlash(at("/canvasfoo"), "/canvas/")).toBeNull();
  });

  it("does nothing without a base path", () => {
    expect(withBasePathSlash(at("/"), "/")).toBeNull();
    expect(withBasePathSlash(at(""), "/")).toBeNull();
  });
});
