import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { HorizontalScrollFade } from "#/components/shared/horizontal-scroll-fade";

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

function renderScrollFade({ draggable = true } = {}) {
  const onClick = vi.fn();
  render(
    <HorizontalScrollFade
      draggable={draggable}
      scrollTestId="scroller"
      fadeTestIdPrefix="fade"
    >
      <div>
        <button type="button" data-testid="chip" onClick={onClick} />
      </div>
    </HorizontalScrollFade>,
  );
  return {
    scroller: screen.getByTestId("scroller"),
    chip: screen.getByTestId("chip"),
    onClick,
  };
}

describe("HorizontalScrollFade", () => {
  it("shows an edge fade only on the clipped side", () => {
    // Arrange
    const { scroller } = renderScrollFade({ draggable: false });
    const leftFade = screen.getByTestId("fade-left");
    const rightFade = screen.getByTestId("fade-right");

    // Act
    mockScrollMetrics(scroller, {
      scrollWidth: 900,
      clientWidth: 320,
      scrollLeft: 0,
    });
    fireEvent.scroll(scroller);

    // Assert
    expect(leftFade).toHaveAttribute("data-visible", "false");
    expect(rightFade).toHaveAttribute("data-visible", "true");

    // Act
    mockScrollMetrics(scroller, {
      scrollWidth: 900,
      clientWidth: 320,
      scrollLeft: 580,
    });
    fireEvent.scroll(scroller);

    // Assert
    expect(leftFade).toHaveAttribute("data-visible", "true");
    expect(rightFade).toHaveAttribute("data-visible", "false");
  });

  describe("draggable", () => {
    it("scrolls while dragging and swallows the click that ends the drag", () => {
      // Arrange
      const { scroller, chip, onClick } = renderScrollFade();
      mockScrollMetrics(scroller, {
        scrollWidth: 900,
        clientWidth: 320,
        scrollLeft: 100,
      });

      // Act
      fireEvent.mouseDown(chip, { button: 0, clientX: 300 });
      fireEvent.mouseMove(document, { clientX: 250, buttons: 1 });
      fireEvent.mouseUp(document);
      fireEvent.click(chip);

      // Assert
      expect(scroller.scrollLeft).toBe(150);
      expect(onClick).not.toHaveBeenCalled();
    });

    it("lets a press without movement click the chip", () => {
      // Arrange
      const { chip, onClick } = renderScrollFade();

      // Act
      fireEvent.mouseDown(chip, { button: 0, clientX: 300 });
      fireEvent.mouseUp(document);
      fireEvent.click(chip);

      // Assert
      expect(onClick).toHaveBeenCalledTimes(1);
    });

    it("cancels the native drag of child images", () => {
      // Arrange
      const { chip } = renderScrollFade();

      // Act
      const notCancelled = fireEvent.dragStart(chip);

      // Assert
      expect(notCancelled).toBe(false);
    });

    it("does not swallow a keyboard click after a drag released outside the content", async () => {
      // Arrange
      const { scroller, chip, onClick } = renderScrollFade();
      mockScrollMetrics(scroller, {
        scrollWidth: 900,
        clientWidth: 320,
        scrollLeft: 100,
      });
      fireEvent.mouseDown(chip, { button: 0, clientX: 300 });
      fireEvent.mouseMove(document, { clientX: 250, buttons: 1 });
      // Released outside the scroller: its click goes elsewhere.
      fireEvent.mouseUp(document);
      await act(async () => {
        await new Promise((resolve) => {
          setTimeout(resolve, 0);
        });
      });

      // Act
      fireEvent.click(chip);

      // Assert
      expect(onClick).toHaveBeenCalledTimes(1);
    });

    it("turns a vertical wheel turn into a horizontal scroll up to the end", () => {
      // Arrange
      const { scroller } = renderScrollFade();
      mockScrollMetrics(scroller, {
        scrollWidth: 900,
        clientWidth: 320,
        scrollLeft: 500,
      });

      // Act
      const notCancelled = fireEvent.wheel(scroller, { deltaY: 100 });

      // Assert: clamped to the end rather than skipped
      expect(notCancelled).toBe(false);
      expect(scroller.scrollLeft).toBe(580);
    });

    it("leaves the wheel to the page once the content is at an edge", () => {
      // Arrange
      const { scroller } = renderScrollFade();
      mockScrollMetrics(scroller, {
        scrollWidth: 900,
        clientWidth: 320,
        scrollLeft: 580,
      });

      // Act
      const notCancelled = fireEvent.wheel(scroller, { deltaY: 100 });

      // Assert
      expect(notCancelled).toBe(true);
      expect(scroller.scrollLeft).toBe(580);
    });

    it("leaves the wheel to the page when the content fits", () => {
      // Arrange
      const { scroller } = renderScrollFade();
      mockScrollMetrics(scroller, {
        scrollWidth: 320,
        clientWidth: 320,
        scrollLeft: 0,
      });

      // Act
      const notCancelled = fireEvent.wheel(scroller, { deltaY: 100 });

      // Assert
      expect(notCancelled).toBe(true);
      expect(scroller.scrollLeft).toBe(0);
    });
  });
});
