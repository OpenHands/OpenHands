import { act, renderHook } from "@testing-library/react";
import type { MouseEvent as ReactMouseEvent } from "react";
import { describe, it, expect, beforeEach } from "vitest";

import { useResizablePanels } from "#/hooks/use-resizable-panels";

describe("useResizablePanels", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("clamps a stale persisted width that is below minLeftWidth on read", () => {
    window.localStorage.setItem("test-panel-width", JSON.stringify(10));

    const { result } = renderHook(() =>
      useResizablePanels({
        defaultLeftWidth: 50,
        minLeftWidth: 30,
        maxLeftWidth: 80,
        storageKey: "test-panel-width",
      }),
    );

    expect(result.current.leftWidth).toBe(30);
    expect(result.current.rightWidth).toBe(70);
  });

  it("clamps a stale persisted width that is above maxLeftWidth on read", () => {
    window.localStorage.setItem("test-panel-width", JSON.stringify(95));

    const { result } = renderHook(() =>
      useResizablePanels({
        defaultLeftWidth: 50,
        minLeftWidth: 30,
        maxLeftWidth: 80,
        storageKey: "test-panel-width",
      }),
    );

    expect(result.current.leftWidth).toBe(80);
    expect(result.current.rightWidth).toBe(20);
  });

  it("uses the persisted value as-is when within range", () => {
    window.localStorage.setItem("test-panel-width", JSON.stringify(60));

    const { result } = renderHook(() =>
      useResizablePanels({
        defaultLeftWidth: 50,
        minLeftWidth: 30,
        maxLeftWidth: 80,
        storageKey: "test-panel-width",
      }),
    );

    expect(result.current.leftWidth).toBe(60);
    expect(result.current.rightWidth).toBe(40);
  });

  it("mounts a full-viewport drag shield while dragging so iframe-heavy panes do not steal pointer events", () => {
    const { result } = renderHook(() =>
      useResizablePanels({
        defaultLeftWidth: 50,
        minLeftWidth: 30,
        maxLeftWidth: 80,
        storageKey: "test-panel-width",
      }),
    );

    expect(document.querySelector("[data-panel-drag-shield]")).toBeNull();

    act(() => {
      result.current.handleMouseDown({
        preventDefault: () => {},
      } as unknown as ReactMouseEvent<HTMLDivElement>);
    });

    const shield = document.querySelector("[data-panel-drag-shield]");
    expect(shield).not.toBeNull();
    expect(shield).toBeInstanceOf(HTMLDivElement);

    act(() => {
      document.dispatchEvent(new MouseEvent("mouseup"));
    });

    expect(document.querySelector("[data-panel-drag-shield]")).toBeNull();
  });
});
