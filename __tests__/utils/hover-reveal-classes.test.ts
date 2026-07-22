import { describe, expect, it, vi, afterEach } from "vitest";
import {
  canHoverReliably,
  fineHoverVariant,
  hoverRevealActionClassName,
  hoverRevealPinnedTimestampClassName,
  hoverRevealReserveClassName,
  hoverRevealYieldClassName,
} from "#/utils/hover-reveal-classes";

describe("hover-reveal-classes", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("keeps overflow actions visible by default and only hides them under fine-hover media", () => {
    const classes = hoverRevealActionClassName();

    expect(classes).toContain("pointer-events-auto");
    expect(classes).toContain("opacity-100");
    expect(classes).toContain(`${fineHoverVariant}pointer-events-none`);
    expect(classes).toContain(`${fineHoverVariant}opacity-0`);
    expect(classes).toContain(
      `${fineHoverVariant}group-hover:pointer-events-auto`,
    );
    expect(classes).toContain(
      `${fineHoverVariant}group-focus-within:opacity-100`,
    );
  });

  it("force-visible actions stay interactable without fine-hover hiding", () => {
    expect(hoverRevealActionClassName(true)).toBe(
      "pointer-events-auto visible opacity-100",
    );
  });

  it("yields timestamp space on touch and restores it under fine-hover media", () => {
    const classes = hoverRevealYieldClassName();

    expect(classes).toContain("opacity-0");
    expect(classes).toContain(`${fineHoverVariant}opacity-100`);
    expect(classes).toContain(`${fineHoverVariant}group-hover:opacity-0`);
  });

  it("reserves action width on touch and collapses it under fine-hover media", () => {
    const classes = hoverRevealReserveClassName();

    expect(classes).toContain("min-w-[3.75rem]");
    expect(classes).toContain(`${fineHoverVariant}min-w-0`);
    expect(classes).toContain(
      `${fineHoverVariant}group-hover:min-w-[3.75rem]`,
    );
  });

  it("hides the pinned-card timestamp under the ellipsis on touch", () => {
    const classes = hoverRevealPinnedTimestampClassName();

    expect(classes).toContain("hidden");
    expect(classes).toContain(`${fineHoverVariant}flex`);
    expect(classes).toContain(`${fineHoverVariant}group-hover:hidden`);
  });

  it("reports canHoverReliably from the fine-hover media query", () => {
    const matchMedia = vi.fn().mockReturnValue({ matches: true });
    vi.stubGlobal("matchMedia", matchMedia);

    expect(canHoverReliably()).toBe(true);
    expect(matchMedia).toHaveBeenCalledWith(
      "(hover: hover) and (pointer: fine)",
    );

    matchMedia.mockReturnValue({ matches: false });
    expect(canHoverReliably()).toBe(false);
  });
});
