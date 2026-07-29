import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
import { code as Code } from "#/components/features/markdown/code";

describe("code (markdown)", () => {
  it("should render inline code without a copy button", () => {
    render(<Code>inline snippet</Code>);

    expect(screen.getByText("inline snippet")).toBeInTheDocument();
    expect(screen.queryByTestId("copy-to-clipboard")).not.toBeInTheDocument();
  });

  it("should render a multiline code block with a copy button", () => {
    render(<Code>{"line1\nline2"}</Code>);

    expect(screen.getByText("line1 line2")).toBeInTheDocument();
    expect(screen.getByTestId("copy-to-clipboard")).toBeInTheDocument();
  });

  it("should render a syntax-highlighted block with a copy button", () => {
    render(<Code className="language-js">{"console.log('hi')"}</Code>);

    expect(screen.getByTestId("copy-to-clipboard")).toBeInTheDocument();
  });

  it("should copy code block content to clipboard", async () => {
    const user = userEvent.setup();
    render(<Code>{"line1\nline2"}</Code>);

    await user.click(screen.getByTestId("copy-to-clipboard"));

    await waitFor(() =>
      expect(navigator.clipboard.readText()).resolves.toBe("line1\nline2"),
    );
  });
});
