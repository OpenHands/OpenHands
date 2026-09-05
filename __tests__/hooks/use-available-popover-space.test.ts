import { renderHook } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";

import { useAvailablePopoverSpace } from "#/hooks/use-available-popover-space";

function makeRef(top: number, bottom: number) {
  const el = {
    getBoundingClientRect: () => ({ top, bottom }) as DOMRect,
  };
  return { current: el } as unknown as React.RefObject<HTMLElement | null>;
}

describe("useAvailablePopoverSpace", () => {
  beforeEach(() => {
    Object.defineProperty(window, "innerHeight", {
      configurable: true,
      value: 800,
    });
    vi.clearAllMocks();
  });

  it("returns undefined while closed", () => {
    const { result } = renderHook(() =>
      useAvailablePopoverSpace(makeRef(500, 540), {
        open: false,
        direction: "up",
      }),
    );
    expect(result.current).toBeUndefined();
  });

  it("caps upward height to the space above the trigger", () => {
    // Trigger sits near the bottom of an 800px viewport — only 300px of room
    // above. The default 480 cap would otherwise clip the top rows off-screen.
    const { result } = renderHook(() =>
      useAvailablePopoverSpace(makeRef(308, 348), {
        open: true,
        direction: "up",
        gap: 8,
      }),
    );
    expect(result.current).toBe(300);
  });

  it("does not exceed the configured maxHeight when there is plenty of room", () => {
    const { result } = renderHook(() =>
      useAvailablePopoverSpace(makeRef(700, 740), {
        open: true,
        direction: "up",
        gap: 8,
        maxHeight: 480,
      }),
    );
    expect(result.current).toBe(480);
  });

  it("measures downward space when direction is down", () => {
    // Trigger near the top: ~700px below it.
    const { result } = renderHook(() =>
      useAvailablePopoverSpace(makeRef(0, 40), {
        open: true,
        direction: "down",
        gap: 8,
      }),
    );
    expect(result.current).toBe(480);
  });
});
