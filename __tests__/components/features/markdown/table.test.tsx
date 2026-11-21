import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import { unified } from "unified";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";

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

  it("parses a long table in time proportional to its rows", () => {
    // Arrange: the GFM parse MarkdownRenderer runs, on a table eight times
    // longer than another. A long table (a listing, a CSV turned into
    // Markdown) is one block, so a streamed reply re-parses it every frame.
    const parser = unified().use(remarkParse).use(remarkGfm);
    const tableWithRows = (rows: number) =>
      `| name | value |\n|---|---|\n${"| `x` | y |\n".repeat(rows)}`;
    const parseTime = (rows: number) => {
      const markdown = tableWithRows(rows);
      const start = performance.now();
      parser.parse(markdown);
      return performance.now() - start;
    };
    parseTime(250); // warm up

    // Act
    const short = parseTime(1000);
    const long = parseTime(8000);

    // Assert: linear parsing takes about 8x as long; quadratic takes 64x.
    expect(long / short).toBeLessThan(20);
  }, 120_000);
});
