import { describe, expect, it } from "vitest";
import {
  EXTENSION_MODULE_CARD_GRID_SINGLE_COLUMN_MAX_PX,
  EXTENSION_MODULE_CARD_INTERACTIVE_CLASS,
  extensionModuleCardGridClassName,
  extensionModuleCardGridContainerClassName,
  extensionModuleCardInteractiveClassName,
  extensionModuleCardSurfaceClassName,
} from "#/utils/extension-module-card-classes";

describe("extensionModuleCardInteractive class", () => {
  it("uses a dedicated CSS class for focus styling", () => {
    expect(EXTENSION_MODULE_CARD_INTERACTIVE_CLASS).toBe(
      "extension-module-card-interactive",
    );
    expect(extensionModuleCardInteractiveClassName).toBe(
      EXTENSION_MODULE_CARD_INTERACTIVE_CLASS,
    );
  });
});

describe("extensionModuleCardSurface class", () => {
  it("omits a resting border so tiles stay flush until hover", () => {
    expect(extensionModuleCardSurfaceClassName).not.toContain("border");
  });
});

describe("extensionModuleCardGrid classes", () => {
  it("uses a container query breakpoint at 600px column width", () => {
    expect(EXTENSION_MODULE_CARD_GRID_SINGLE_COLUMN_MAX_PX).toBe(599);
    expect(extensionModuleCardGridContainerClassName).toContain("@container");
    expect(extensionModuleCardGridClassName).toContain("@min-[600px]:grid-cols-2");
    expect(extensionModuleCardGridClassName).not.toContain("md:grid-cols-2");
  });
});
