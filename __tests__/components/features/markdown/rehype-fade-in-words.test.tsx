import React from "react";
import { render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";
import {
  RevealClock,
  STREAM_WORD_FADE_MS,
} from "#/components/features/markdown/rehype-fade-in-words";

describe("RevealClock", () => {
  it("reports when each part was revealed until its fade is over", () => {
    const clock = new RevealClock();
    clock.advance(5, 0);
    clock.advance(9, 100);

    expect(clock.revealedAt(2)).toBe(0);
    expect(clock.revealedAt(7)).toBe(100);

    clock.advance(12, STREAM_WORD_FADE_MS + 50);
    // The first part has settled; later parts still have a time.
    expect(clock.revealedAt(2)).toBeNull();
    expect(clock.revealedAt(7)).toBe(100);
  });

  it("treats text as shown after a reset", () => {
    const clock = new RevealClock();
    clock.advance(5, 0);
    clock.reset(20);
    expect(clock.revealedAt(10)).toBeNull();
  });
});

describe("streamed words fading in", () => {
  let now = 0;

  beforeEach(() => {
    now = 0;
    vi.spyOn(performance, "now").mockImplementation(() => now);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  const word = (container: HTMLElement, text: string) =>
    [...container.querySelectorAll<HTMLElement>("[data-stream-word]")].find(
      (span) => span.textContent === text,
    );

  it("fades in new words and leaves settled ones alone", () => {
    const { container, rerender } = render(
      <MarkdownRenderer streaming>{"Hello"}</MarkdownRenderer>,
    );
    expect(word(container, "Hello")).toHaveClass("stream-word-fade");

    now = 1000;
    rerender(<MarkdownRenderer streaming>{"Hello world"}</MarkdownRenderer>);
    expect(word(container, "Hello")).not.toHaveClass("stream-word-fade");
    expect(word(container, "world")).toHaveClass("stream-word-fade");
    expect(word(container, "world")?.style.animationDelay).toBe("0ms");
  });

  it("resumes a word's fade when its block is re-created", () => {
    const { container, rerender } = render(
      <MarkdownRenderer streaming>{"One.\n\nTwo.\n\nThr"}</MarkdownRenderer>,
    );

    // "Two." freezes, so the tail holding "Three." is a new block.
    now = 100;
    rerender(
      <MarkdownRenderer streaming>
        {"One.\n\nTwo.\n\nThree.\n\nFou"}
      </MarkdownRenderer>,
    );

    // Revealed (as "Thr") 100ms ago: it continues, not restarts, its fade.
    expect(word(container, "Three.")?.style.animationDelay).toBe("-100ms");
  });

  it("does not split code", () => {
    const { container } = render(
      <MarkdownRenderer streaming>
        {"Run `npm test` now\n\n```sh\nnpm run build\n```"}
      </MarkdownRenderer>,
    );
    expect(word(container, "Run")).toBeDefined();
    expect(container.querySelector("code [data-stream-word]")).toBeNull();
  });

  it("adds nothing to a finished message", () => {
    const { container } = render(
      <MarkdownRenderer>{"Hello world"}</MarkdownRenderer>,
    );
    expect(container.querySelector("[data-stream-word]")).toBeNull();
  });
});
