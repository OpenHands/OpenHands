import { describe, expect, it, vi } from "vitest";

import { createCommandMenuItems } from "#/components/features/command-menu/command-menu-items";

/**
 * The automation entry's title, description, and keywords are the interface
 * manifest's. Without one there is no copy to show and no surface to reach,
 * so the entry is not offered at all.
 */
vi.mock("#/manifests/manifest-sources", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/manifests/manifest-sources")>();
  return { ...actual, AUTOMATION_INTERFACE_CANDIDATE: undefined };
});

describe("the command menu without an admitted interface manifest", () => {
  it("omits the automations entry and keeps the rest", () => {
    // Act
    const items = createCommandMenuItems({ toggleSidebar: vi.fn() });

    // Assert
    expect(items.some((item) => item.id === "automations")).toBe(false);
    expect(items.some((item) => item.id === "new-chat")).toBe(true);
  });
});
