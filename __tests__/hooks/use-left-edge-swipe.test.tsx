import type { ReactNode } from "react";
import { createPortal } from "react-dom";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useLeftEdgeSwipe } from "#/hooks/use-left-edge-swipe";

const onSwipe = vi.fn();

function SwipeSurface({
  enabled = true,
  children,
}: {
  enabled?: boolean;
  children: ReactNode;
}) {
  const handlers = useLeftEdgeSwipe({ enabled, onSwipe });
  return (
    <div data-testid="surface" {...handlers}>
      {children}
    </div>
  );
}

type Point = [x: number, y: number];

function touchAt([clientX, clientY]: Point, identifier = 0) {
  return { identifier, clientX, clientY };
}

/** Touch down at `from`, move through `moves`, and lift at the last point. */
function swipe(element: Element, from: Point, ...moves: Point[]) {
  fireEvent.touchStart(element, { touches: [touchAt(from)] });
  for (const point of moves) {
    fireEvent.touchMove(element, {
      touches: [touchAt(point)],
      changedTouches: [touchAt(point)],
    });
  }
  const end = moves.at(-1) ?? from;
  fireEvent.touchEnd(element, { touches: [], changedTouches: [touchAt(end)] });
}

/** Mimic a scroll container whose content overflows horizontally or not. */
function makeScrollable(element: HTMLElement, overflowsHorizontally: boolean) {
  // Browsers compute overflow-x to auto once either axis scrolls.
  element.style.setProperty("overflow-x", "auto");
  element.style.setProperty("overflow-y", "auto");
  Object.defineProperties(element, {
    clientWidth: { configurable: true, value: 300 },
    scrollWidth: {
      configurable: true,
      value: overflowsHorizontally ? 900 : 300,
    },
  });
}

function PortaledTarget() {
  return createPortal(<div data-testid="target" />, document.body);
}

describe("useLeftEdgeSwipe", () => {
  beforeEach(() => {
    onSwipe.mockClear();
  });

  it("fires once for a rightward swipe from the left edge inside a vertical scroller", () => {
    render(
      <SwipeSurface>
        <div data-testid="message-list">
          <p data-testid="target">message</p>
        </div>
      </SwipeSurface>,
    );
    makeScrollable(screen.getByTestId("message-list"), false);

    swipe(
      screen.getByTestId("target"),
      [8, 300],
      [40, 304],
      [90, 310],
      [160, 312],
    );

    expect(onSwipe).toHaveBeenCalledOnce();
  });

  it("fires on release when the browser coalesces the whole swipe", () => {
    render(
      <SwipeSurface>
        <p data-testid="target">message</p>
      </SwipeSurface>,
    );
    const target = screen.getByTestId("target");

    fireEvent.touchStart(target, { touches: [touchAt([4, 200])] });
    fireEvent.touchEnd(target, {
      touches: [],
      changedTouches: [touchAt([120, 210])],
    });

    expect(onSwipe).toHaveBeenCalledOnce();
  });

  it.each<[string, Point, ...Point[]]>([
    ["starts away from the edge", [80, 300], [200, 300]],
    ["is too short", [8, 300], [50, 300]],
    ["scrolls vertically first", [8, 300], [10, 250], [120, 240]],
    ["is mostly diagonal", [8, 300], [90, 240]],
    ["moves leftward", [24, 300], [0, 300]],
  ])("does not fire for a touch that %s", (_name, from, ...moves) => {
    render(
      <SwipeSurface>
        <p data-testid="target">message</p>
      </SwipeSurface>,
    );

    swipe(screen.getByTestId("target"), from, ...moves);

    expect(onSwipe).not.toHaveBeenCalled();
  });

  it("does not fire while disabled", () => {
    render(
      <SwipeSurface enabled={false}>
        <p data-testid="target">message</p>
      </SwipeSurface>,
    );

    swipe(screen.getByTestId("target"), [8, 300], [160, 300]);

    expect(onSwipe).not.toHaveBeenCalled();
  });

  it("cancels when a second finger joins", () => {
    render(
      <SwipeSurface>
        <p data-testid="target">message</p>
      </SwipeSurface>,
    );
    const target = screen.getByTestId("target");

    fireEvent.touchStart(target, { touches: [touchAt([8, 300])] });
    fireEvent.touchMove(target, {
      touches: [touchAt([40, 300]), touchAt([200, 100], 1)],
    });
    fireEvent.touchEnd(target, {
      touches: [],
      changedTouches: [touchAt([160, 300])],
    });

    expect(onSwipe).not.toHaveBeenCalled();
  });

  it.each<[string, () => ReactNode]>([
    ["a text input", () => <input data-testid="target" aria-label="Name" />],
    ["a textarea", () => <textarea data-testid="target" aria-label="Notes" />],
    [
      "a contenteditable composer",
      () => (
        <div contentEditable suppressContentEditableWarning>
          <span data-testid="target">draft</span>
        </div>
      ),
    ],
    [
      "a horizontal scroller",
      () => (
        <div data-testid="scroller">
          <code data-testid="target">long line</code>
        </div>
      ),
    ],
    ["portaled content", () => <PortaledTarget />],
  ])("leaves touches on %s to their native behaviour", (_name, content) => {
    render(<SwipeSurface>{content()}</SwipeSurface>);
    const scroller = screen.queryByTestId("scroller");
    if (scroller) makeScrollable(scroller, true);

    swipe(screen.getByTestId("target"), [8, 300], [160, 300]);

    expect(onSwipe).not.toHaveBeenCalled();
  });
});
