import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
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
