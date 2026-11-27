import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { EditorContainer } from "./editor-container";

describe("EditorContainer", () => {
  it("renders children", () => {
    render(
      <EditorContainer height={400}>
        <div data-testid="child" />
      </EditorContainer>,
    );
    expect(screen.getByTestId("child")).toBeInTheDocument();
  });

  it("renders with the editor-container test id", () => {
    render(
      <EditorContainer height={300}>
        <div data-testid="inner" />
      </EditorContainer>,
    );
    expect(screen.getByTestId("editor-container")).toBeInTheDocument();
  });

  // Functional contract: the `height` prop is observable only via the
  // `--editor-height` CSS custom property the component writes to the DOM.
  it("sets the --editor-height CSS variable based on height prop", () => {
    render(
      <EditorContainer height={500}>
        <div data-testid="inner" />
      </EditorContainer>,
    );
    const container = screen.getByTestId("editor-container");
    expect(container.style.getPropertyValue("--editor-height")).toBe("500px");
  });
});
