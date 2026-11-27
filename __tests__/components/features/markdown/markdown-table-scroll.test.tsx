import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  MarkdownTableScroll,
  readScrollFadeState,
} from "#/components/features/markdown/markdown-table-scroll";

function mockScrollMetrics(
  element: HTMLElement,
  metrics: { scrollWidth: number; clientWidth: number; scrollLeft: number },
) {
  Object.defineProperty(element, "scrollWidth", {
    configurable: true,
    value: metrics.scrollWidth,
  });
  Object.defineProperty(element, "clientWidth", {
    configurable: true,
    value: metrics.clientWidth,
  });
  Object.defineProperty(element, "scrollLeft", {
    configurable: true,
    writable: true,
    value: metrics.scrollLeft,
  });
}

describe("readScrollFadeState", () => {
  it("hides both fades when the table fits", () => {
    const element = document.createElement("div");
    mockScrollMetrics(element, {
      scrollWidth: 400,
      clientWidth: 400,
      scrollLeft: 0,
    });

    expect(readScrollFadeState(element)).toEqual({ left: false, right: false });
  });

  it("shows only the right fade at the start of an overflowing table", () => {
    const element = document.createElement("div");
    mockScrollMetrics(element, {
      scrollWidth: 800,
      clientWidth: 300,
      scrollLeft: 0,
    });

    expect(readScrollFadeState(element)).toEqual({ left: false, right: true });
  });

  it("shows both fades in the middle of an overflowing table", () => {
    const element = document.createElement("div");
    mockScrollMetrics(element, {
      scrollWidth: 800,
      clientWidth: 300,
      scrollLeft: 250,
    });

    expect(readScrollFadeState(element)).toEqual({ left: true, right: true });
  });

  it("shows only the left fade at the end of an overflowing table", () => {
    const element = document.createElement("div");
    mockScrollMetrics(element, {
      scrollWidth: 800,
      clientWidth: 300,
      scrollLeft: 500,
    });

    expect(readScrollFadeState(element)).toEqual({ left: true, right: false });
  });
});

describe("MarkdownTableScroll", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "ResizeObserver",
      class {
        observe = vi.fn();

        unobserve = vi.fn();

        disconnect = vi.fn();
      },
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders animated edge fades that toggle with horizontal scroll", () => {
    render(
      <MarkdownTableScroll>
        <table>
          <tbody>
            <tr>
              <td>Wide content</td>
            </tr>
          </tbody>
        </table>
      </MarkdownTableScroll>,
    );

    const scroller = screen.getByTestId("markdown-table-scroll");
    const leftFade = screen.getByTestId("markdown-table-scroll-fade-left");
    const rightFade = screen.getByTestId("markdown-table-scroll-fade-right");

    mockScrollMetrics(scroller, {
      scrollWidth: 900,
      clientWidth: 320,
      scrollLeft: 0,
    });
    fireEvent.scroll(scroller);

    expect(rightFade).toHaveAttribute("data-visible", "true");
    expect(rightFade).toHaveClass("opacity-100");
    expect(leftFade).toHaveAttribute("data-visible", "false");
    expect(leftFade).toHaveClass("opacity-0");

    mockScrollMetrics(scroller, {
      scrollWidth: 900,
      clientWidth: 320,
      scrollLeft: 580,
    });
    fireEvent.scroll(scroller);

    expect(leftFade).toHaveAttribute("data-visible", "true");
    expect(leftFade).toHaveClass("opacity-100");
    expect(rightFade).toHaveAttribute("data-visible", "false");
    expect(rightFade).toHaveClass("opacity-0");
  });
});
