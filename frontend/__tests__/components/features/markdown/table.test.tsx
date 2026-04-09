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
    // border-collapse + border is what makes columns visually separate
    expect(table).toHaveClass("border-collapse");
    expect(table).toHaveClass("border");
  });

  it("should wrap the table in a horizontally scrollable container", () => {
    const { container } = render(
      <MarkdownRenderer>{GFM_TABLE}</MarkdownRenderer>,
    );

    // Wide tables must not break chat layout — wrapper enables overflow
    const wrapper = container.querySelector(".overflow-x-auto");
    expect(wrapper).not.toBeNull();
    expect(wrapper?.querySelector("table")).not.toBeNull();
  });

  it("should render header cells as styled <th> elements", () => {
    render(<MarkdownRenderer>{GFM_TABLE}</MarkdownRenderer>);

    const headers = screen.getAllByRole("columnheader");
    expect(headers).toHaveLength(3);
    expect(headers[0]).toHaveTextContent("Feature");
    expect(headers[1]).toHaveTextContent("OpenAI Codex");
    expect(headers[2]).toHaveTextContent("Claude Code");
    // Padding + border is what was missing before the fix
    headers.forEach((h) => {
      expect(h).toHaveClass("border");
      expect(h).toHaveClass("px-3");
      expect(h).toHaveClass("py-2");
    });
  });

  it("should render body cells as styled <td> elements", () => {
    render(<MarkdownRenderer>{GFM_TABLE}</MarkdownRenderer>);

    const cells = screen.getAllByRole("cell");
    expect(cells).toHaveLength(6);
    expect(cells[0]).toHaveTextContent("CLI");
    expect(cells[3]).toHaveTextContent("Mobile");
    cells.forEach((c) => {
      expect(c).toHaveClass("border");
      expect(c).toHaveClass("px-3");
      expect(c).toHaveClass("py-2");
    });
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
