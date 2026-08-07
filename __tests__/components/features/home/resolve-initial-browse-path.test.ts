import { describe, expect, it } from "vitest";
import { resolveInitialBrowsePath } from "#/components/features/home/workspace-dropdown/folder-browser-modal";

describe("resolveInitialBrowsePath", () => {
  it("prefers the configured default workspace browse path", () => {
    expect(
      resolveInitialBrowsePath(
        { home: "/home/user", favorites: [], locations: [] },
        "/data/workspaces",
      ),
    ).toBe("/data/workspaces");
  });

  it("falls back to /projects for the Docker openhands home", () => {
    expect(
      resolveInitialBrowsePath(
        { home: "/home/openhands", favorites: [], locations: [] },
        null,
      ),
    ).toBe("/projects");
  });

  it("falls back to the server home when no preference is set", () => {
    expect(
      resolveInitialBrowsePath(
        { home: "/Users/me", favorites: [], locations: [] },
        "   ",
      ),
    ).toBe("/Users/me");
  });
});
