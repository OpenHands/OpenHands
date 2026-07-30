import { describe, expect, it } from "vitest";
import {
  POPOUT_DOCK_INSET_PX,
  POPOUT_EXPANDED_WIDTH_PX,
  POPOUT_GAP_PX,
  POPOUT_MINIMIZED_WIDTH_PX,
  POPOUT_OVERFLOW_BUTTON_SIZE_PX,
  type Popout,
} from "#/stores/popout-store";
import { layoutPopouts } from "#/components/features/popout/popout-layout";

const makePopout = (
  id: string,
  openedAt: number,
  mode: Popout["mode"] = "expanded",
): Popout => ({
  conversationId: id,
  title: id,
  prefillMessage: null,
  mode,
  openedAt,
});

describe("layoutPopouts", () => {
  it("keeps every popout visible in open order when they fit", () => {
    const popouts = [makePopout("a", 1), makePopout("b", 2)];
    // Two expanded popouts + gap + insets fit in a wide viewport.
    const width =
      POPOUT_DOCK_INSET_PX * 2 + POPOUT_EXPANDED_WIDTH_PX * 2 + POPOUT_GAP_PX;

    const layout = layoutPopouts(popouts, width);

    expect(layout.visible.map((entry) => entry.conversationId)).toEqual([
      "a",
      "b",
    ]);
    expect(layout.hidden).toEqual([]);
  });

  it("hides the least recently active popouts and reserves the overflow button", () => {
    const popouts = [
      makePopout("oldest", 1),
      makePopout("middle", 2),
      makePopout("newest", 3),
    ];
    // Only room for one expanded popout and the overflow count button.
    const width =
      POPOUT_DOCK_INSET_PX * 2 +
      POPOUT_EXPANDED_WIDTH_PX +
      POPOUT_OVERFLOW_BUTTON_SIZE_PX +
      POPOUT_GAP_PX;

    const layout = layoutPopouts(popouts, width);

    expect(layout.visible.map((entry) => entry.conversationId)).toEqual([
      "newest",
    ]);
    expect(layout.hidden.map((entry) => entry.conversationId)).toEqual([
      "middle",
      "oldest",
    ]);
  });

  it("uses the smaller footprint of an explicitly minimized popout", () => {
    const popouts = [
      makePopout("kept-min", 1, "minimized"),
      makePopout("expanded", 2, "expanded"),
    ];
    const width =
      POPOUT_DOCK_INSET_PX * 2 +
      POPOUT_EXPANDED_WIDTH_PX +
      POPOUT_MINIMIZED_WIDTH_PX +
      POPOUT_GAP_PX;

    const layout = layoutPopouts(popouts, width);

    expect(layout.visible).toEqual(popouts);
    expect(layout.hidden).toEqual([]);
  });

  it("promotes a reopened hidden popout and displaces the least recent visible one", () => {
    const popouts = [
      makePopout("formerly-visible", 2),
      makePopout("reopened", 3),
      makePopout("least-recent", 1),
    ];
    const width =
      POPOUT_DOCK_INSET_PX * 2 +
      POPOUT_EXPANDED_WIDTH_PX * 2 +
      POPOUT_OVERFLOW_BUTTON_SIZE_PX +
      POPOUT_GAP_PX * 2;

    const layout = layoutPopouts(popouts, width);

    expect(layout.visible.map((entry) => entry.conversationId)).toEqual([
      "formerly-visible",
      "reopened",
    ]);
    expect(layout.hidden.map((entry) => entry.conversationId)).toEqual([
      "least-recent",
    ]);
  });

  it("treats explicitly minimized popouts as less active than expanded ones", () => {
    const popouts = [
      makePopout("older-expanded", 1),
      makePopout("newer-minimized", 2, "minimized"),
    ];
    const width =
      POPOUT_DOCK_INSET_PX * 2 +
      POPOUT_EXPANDED_WIDTH_PX +
      POPOUT_OVERFLOW_BUTTON_SIZE_PX +
      POPOUT_GAP_PX;

    const layout = layoutPopouts(popouts, width);

    expect(layout.visible.map((entry) => entry.conversationId)).toEqual([
      "older-expanded",
    ]);
    expect(layout.hidden.map((entry) => entry.conversationId)).toEqual([
      "newer-minimized",
    ]);
  });
});
