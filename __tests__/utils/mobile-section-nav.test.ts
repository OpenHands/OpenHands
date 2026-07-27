import { describe, expect, it } from "vitest";
import { getMobileTopBarState } from "#/utils/mobile-section-nav";
import { I18nKey } from "#/i18n/declaration";

describe("getMobileTopBarState", () => {
  it("shows menu on settings and customize hubs", () => {
    expect(getMobileTopBarState("/settings")).toEqual({ mode: "menu" });
    expect(getMobileTopBarState("/customize")).toEqual({ mode: "menu" });
  });

  it("backs from settings detail pages to the settings hub", () => {
    expect(getMobileTopBarState("/settings/llm")).toEqual({
      mode: "back",
      backTo: "/settings",
      backLabelKey: I18nKey.SETTINGS$TITLE,
    });
  });

  it("backs from extension detail pages to the customize hub", () => {
    expect(getMobileTopBarState("/skills")).toEqual({
      mode: "back",
      backTo: "/customize",
      backLabelKey: I18nKey.NAV$CUSTOMIZE,
    });
    expect(getMobileTopBarState("/mcp")).toEqual({
      mode: "back",
      backTo: "/customize",
      backLabelKey: I18nKey.NAV$CUSTOMIZE,
    });
  });

  it("shows menu on main app routes", () => {
    expect(getMobileTopBarState("/conversations")).toEqual({ mode: "menu" });
  });
});
