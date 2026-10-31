import React from "react";
import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";
import {
  RevealClock,
  STREAM_WORD_FADE_MS,
} from "#/components/features/markdown/rehype-fade-in-words";

describe("rehype-fade-in-words", () => {
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
});

describe("table", () => {
  const GFM_TABLE = [
    "| Feature | OpenAI Codex | Claude Code |",
    "|---------|--------------|-------------|",
    "| CLI     | ✅           | ✅          |",
    "| Mobile  | ❌           | ✅          |",
  ].join("\n");

  describe("table (markdown)", () => {
    it("should render a GFM pipe table as a <table> element", () => {
      render(<MarkdownRenderer>{GFM_TABLE}</MarkdownRenderer>);

      const table = screen.getByRole("table");
      expect(table).toBeInTheDocument();
    });

    it("should wrap the table in a horizontally scrollable container", () => {
      const { container } = render(
        <MarkdownRenderer>{GFM_TABLE}</MarkdownRenderer>,
      );

      const wrapper = screen.getByTestId("markdown-table-scroll");
      expect(wrapper).toHaveClass("overflow-x-auto");
      expect(wrapper).toHaveClass("custom-scrollbar-always");
      expect(wrapper).toHaveClass("max-w-full");
      expect(wrapper.querySelector("table")).not.toBeNull();
      expect(container.querySelector(".overflow-x-auto")).not.toBeNull();
    });

    it("should round corners on the table element", () => {
      render(<MarkdownRenderer>{GFM_TABLE}</MarkdownRenderer>);

      const table = screen.getByRole("table");
      expect(table).toHaveClass("rounded-xl");
      expect(table).toHaveClass("overflow-hidden");

      const wrapper = screen.getByTestId("markdown-table-scroll");
      expect(wrapper).not.toHaveClass("rounded-xl");

      expect(
        screen.getByTestId("markdown-table-scroll-fade-left"),
      ).toBeInTheDocument();
      expect(
        screen.getByTestId("markdown-table-scroll-fade-right"),
      ).toBeInTheDocument();
    });

    it("should let wide tables grow beyond the chat column instead of squishing columns", () => {
      render(<MarkdownRenderer>{GFM_TABLE}</MarkdownRenderer>);

      const table = screen.getByRole("table");
      expect(table).toHaveClass("w-max");
      expect(table).toHaveClass("min-w-full");
      expect(table).not.toHaveClass("w-full");
    });

    it("should cap cell width and wrap long text instead of stretching the table", () => {
      render(<MarkdownRenderer>{GFM_TABLE}</MarkdownRenderer>);

      [
        ...screen.getAllByRole("columnheader"),
        ...screen.getAllByRole("cell"),
      ].forEach((cell) => {
        expect(cell).toHaveClass("max-w-md");
        expect(cell).toHaveClass("break-words");
        expect(cell).not.toHaveClass("whitespace-nowrap");
      });
    });

    it("should render header cells as <th> elements with correct content", () => {
      render(<MarkdownRenderer>{GFM_TABLE}</MarkdownRenderer>);

      const headers = screen.getAllByRole("columnheader");
      expect(headers).toHaveLength(3);
      expect(headers[0]).toHaveTextContent("Feature");
      expect(headers[1]).toHaveTextContent("OpenAI Codex");
      expect(headers[2]).toHaveTextContent("Claude Code");
    });

    it("should render body cells as <td> elements with correct content", () => {
      render(<MarkdownRenderer>{GFM_TABLE}</MarkdownRenderer>);

      const cells = screen.getAllByRole("cell");
      expect(cells).toHaveLength(6);
      expect(cells[0]).toHaveTextContent("CLI");
      expect(cells[3]).toHaveTextContent("Mobile");
    });

    it("should not render table markdown as plain paragraph text", () => {
      // Regression guard: before the fix, missing component overrides made the
      // table render with no visible borders/padding so columns looked like
      // space-separated text. Ensure a real <table> exists now.
      const { container } = render(
        <MarkdownRenderer>{GFM_TABLE}</MarkdownRenderer>,
      );

      expect(container.querySelectorAll("table")).toHaveLength(1);
      expect(container.querySelectorAll("th")).toHaveLength(3);
      expect(container.querySelectorAll("td")).toHaveLength(6);
    });
  });
});
