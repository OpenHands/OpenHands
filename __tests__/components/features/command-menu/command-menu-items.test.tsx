import { afterEach, describe, expect, it, vi } from "vitest";

import { createCommandMenuItems } from "#/components/features/command-menu/command-menu-items";
import { OSS_NAV_ITEMS } from "#/constants/settings-nav";

function createLocalCommandMenuItems({
  includeDesktopSidebarToggle = true,
  toggleSidebar = vi.fn(),
}: {
  includeDesktopSidebarToggle?: boolean;
  toggleSidebar?: () => void;
} = {}) {
  return createCommandMenuItems({
    toggleSidebar,
    backendKind: "local",
    includeDesktopSidebarToggle,
  });
}

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
    const items = createLocalCommandMenuItems();

    // Assert
    expect(items.some((item) => item.id === "automations")).toBe(false);
    expect(items.some((item) => item.id === "new-chat")).toBe(true);
  });
});

const settingsItemIds = (items: ReturnType<typeof createCommandMenuItems>) =>
  items.filter((item) => item.group === "settings").map((item) => item.id);

/**
 * Locked to a Cloud host (SaaS / self-hosted OHE) the settings sidebar lists
 * only the Application page; the command menu must not re-expose the pages
 * that sidebar unlists.
 */
describe("the command menu settings group", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("lists every settings page when not locked to a Cloud host", () => {
    // Act
    const items = createLocalCommandMenuItems();

    // Assert
    expect(settingsItemIds(items)).toEqual([
      "settings",
      "agent-settings",
      "agent-context-settings",
      "verification-settings",
      "runtime-settings",
      "app-settings",
      "experimental-settings",
    ]);
  });

  it("offers an entry for every page the settings nav lists", () => {
    // Act
    const routes = createLocalCommandMenuItems().map((item) => item.to);

    // Assert — Condenser is per agent profile on the default local backend.
    expect(routes).toEqual(
      expect.arrayContaining(
        OSS_NAV_ITEMS.map((navItem) => navItem.to).filter(
          (to) => to !== "/settings/condenser" && to !== "/settings/meta-llm",
        ),
      ),
    );
  });

  it("keeps only Settings and Application when locked to a Cloud host", () => {
    // Arrange
    vi.stubEnv("VITE_LOCK_TO_CLOUD", "https://app.all-hands.dev");

    // Act
    const items = createLocalCommandMenuItems();

    // Assert
    expect(settingsItemIds(items)).toEqual(["settings", "app-settings"]);
    expect(items.map((item) => item.id)).toEqual([
      "new-chat",
      "customize",
      "settings",
      "app-settings",
      "toggle-sidebar",
    ]);
  });
});

describe("the command menu sidebar action", () => {
  it("omits the desktop sidebar toggle when the sidebar rail is hidden", () => {
    // Arrange
    const toggleSidebar = vi.fn();

    // Act
    const items = createLocalCommandMenuItems({
      includeDesktopSidebarToggle: false,
      toggleSidebar,
    });

    // Assert
    expect(items.find((item) => item.id === "toggle-sidebar")).toBeUndefined();
    expect(toggleSidebar).not.toHaveBeenCalled();
  });

  it("keeps the desktop sidebar toggle when the sidebar rail is visible", () => {
    // Arrange
    const toggleSidebar = vi.fn();

    // Act
    const items = createLocalCommandMenuItems({
      includeDesktopSidebarToggle: true,
      toggleSidebar,
    });
    const toggleItem = items.find((item) => item.id === "toggle-sidebar");
    toggleItem?.perform?.();

    // Assert
    expect(toggleItem).toBeDefined();
    expect(toggleSidebar).toHaveBeenCalledOnce();
  });
});
