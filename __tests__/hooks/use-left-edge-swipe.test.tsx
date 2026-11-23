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

/** A phone-width surface: the swipe zone runs from 24px to 130px. */
function renderSurface(ui: ReactNode) {
  render(ui);
  vi.spyOn(
    screen.getByTestId("surface"),
    "getBoundingClientRect",
  ).mockReturnValue({ left: 0, top: 0, width: 390, height: 844 } as DOMRect);
}

function PortaledTarget() {
  return createPortal(<div data-testid="target" />, document.body);
}

describe("useLeftEdgeSwipe", () => {
  beforeEach(() => {
    onSwipe.mockClear();
  });

  it("fires once for a rightward swipe from the left third inside a vertical scroller", () => {
    renderSurface(
      <SwipeSurface>
        <div data-testid="message-list">
          <p data-testid="target">message</p>
        </div>
      </SwipeSurface>,
    );
    makeScrollable(screen.getByTestId("message-list"), false);

    swipe(
      screen.getByTestId("target"),
      [40, 300],
      [70, 304],
      [120, 310],
      [200, 312],
    );

    expect(onSwipe).toHaveBeenCalledOnce();
  });

  it.each<[string, Point, Point]>([
    ["at the inner edge of the OS gutter", [24, 300], [160, 304]],
    ["at the end of the left third", [130, 300], [260, 304]],
  ])("fires for a swipe that starts %s", (_name, from, to) => {
    renderSurface(
      <SwipeSurface>
        <p data-testid="target">message</p>
      </SwipeSurface>,
    );

    swipe(screen.getByTestId("target"), from, to);

    expect(onSwipe).toHaveBeenCalledOnce();
  });

  it("fires on release when the browser coalesces the whole swipe", () => {
    renderSurface(
      <SwipeSurface>
        <p data-testid="target">message</p>
      </SwipeSurface>,
    );
    const target = screen.getByTestId("target");

    fireEvent.touchStart(target, { touches: [touchAt([30, 200])] });
    fireEvent.touchEnd(target, {
      touches: [],
      changedTouches: [touchAt([150, 210])],
    });

    expect(onSwipe).toHaveBeenCalledOnce();
  });

  it.each<[string, Point, ...Point[]]>([
    ["starts in the OS edge gutter", [23, 300], [200, 300]],
    ["starts beyond the left third", [131, 300], [300, 300]],
    ["is too short", [40, 300], [90, 300]],
    ["scrolls vertically first", [40, 300], [42, 250], [160, 240]],
    ["is mostly diagonal", [40, 300], [120, 240]],
    ["moves leftward", [100, 300], [40, 300]],
  ])("does not fire for a touch that %s", (_name, from, ...moves) => {
    renderSurface(
      <SwipeSurface>
        <p data-testid="target">message</p>
      </SwipeSurface>,
    );

    swipe(screen.getByTestId("target"), from, ...moves);

    expect(onSwipe).not.toHaveBeenCalled();
  });

  it("does not fire while disabled", () => {
    renderSurface(
      <SwipeSurface enabled={false}>
        <p data-testid="target">message</p>
      </SwipeSurface>,
    );

    swipe(screen.getByTestId("target"), [40, 300], [200, 300]);

    expect(onSwipe).not.toHaveBeenCalled();
  });

  it("cancels when a second finger joins", () => {
    renderSurface(
      <SwipeSurface>
        <p data-testid="target">message</p>
      </SwipeSurface>,
    );
    const target = screen.getByTestId("target");

    fireEvent.touchStart(target, { touches: [touchAt([40, 300])] });
    fireEvent.touchMove(target, {
      touches: [touchAt([70, 300]), touchAt([200, 100], 1)],
    });
    fireEvent.touchEnd(target, {
      touches: [],
      changedTouches: [touchAt([200, 300])],
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
    renderSurface(<SwipeSurface>{content()}</SwipeSurface>);
    const scroller = screen.queryByTestId("scroller");
    if (scroller) makeScrollable(scroller, true);

    swipe(screen.getByTestId("target"), [40, 300], [200, 300]);

    expect(onSwipe).not.toHaveBeenCalled();
  });
});
