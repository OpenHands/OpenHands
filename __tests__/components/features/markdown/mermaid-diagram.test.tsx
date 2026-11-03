import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { code as Code } from "#/components/features/markdown/code";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";

const mermaidMock = vi.hoisted(() => ({
  initialize: vi.fn(),
  parse: vi.fn(),
  render: vi.fn(),
}));

vi.mock("mermaid", () => ({ default: mermaidMock }));

// Rendered SVGs are cached per source for the module's lifetime, so each test
// that needs a fresh render uses its own source.
const SOURCE = "flowchart LR\n  A --> B";
const MARKDOWN_SOURCE = "flowchart LR\n  M --> N";
const PENDING_SOURCE = "flowchart LR\n  P --> Q";
const CACHED_SOURCE = "flowchart LR\n  C --> D";
const INVALID_SOURCE = "flowchart LR\n  A -->";
const PYTHON_SOURCE = "print('hi')";

describe("mermaid code blocks", () => {
  beforeEach(() => {
    mermaidMock.initialize.mockReset();
    mermaidMock.parse
      .mockReset()
      .mockResolvedValue({ diagramType: "flowchart" });
    mermaidMock.render
      .mockReset()
      .mockResolvedValue({ svg: '<svg data-testid="rendered-svg"></svg>' });
  });

  it("renders a mermaid fence as a diagram in strict security mode", async () => {
    render(<Code className="language-mermaid">{SOURCE}</Code>);

    expect(await screen.findByTestId("mermaid-diagram")).toContainElement(
      screen.getByTestId("rendered-svg"),
    );
    expect(mermaidMock.initialize).toHaveBeenCalledWith(
      expect.objectContaining({ securityLevel: "strict", startOnLoad: false }),
    );
    expect(mermaidMock.render).toHaveBeenCalledWith(
      expect.stringMatching(/^mermaid-\d+$/),
      SOURCE,
    );
  });

  it("keeps the code block when the source does not parse", async () => {
    mermaidMock.parse.mockResolvedValue(false);

    const { container } = render(
      <Code className="language-mermaid">{INVALID_SOURCE}</Code>,
    );

    await waitFor(() => expect(mermaidMock.parse).toHaveBeenCalled());
    expect(mermaidMock.render).not.toHaveBeenCalled();
    expect(screen.queryByTestId("mermaid-diagram")).not.toBeInTheDocument();
    expect(container).toHaveTextContent("A -->");
  });

  it("renders fenced mermaid blocks inside markdown", async () => {
    render(
      <MarkdownRenderer
        content={`Intro text\n\n\`\`\`mermaid\n${MARKDOWN_SOURCE}\n\`\`\``}
      />,
    );

    expect(screen.getByText("Intro text")).toBeInTheDocument();
    expect(await screen.findByTestId("mermaid-diagram")).toBeInTheDocument();
  });

  it("holds the space with a placeholder, not the source, while rendering", async () => {
    // Arrange
    let finishRender: (value: { svg: string }) => void = () => {};
    mermaidMock.render.mockReturnValue(
      new Promise((resolve) => {
        finishRender = resolve;
      }),
    );

    // Act
    const { container } = render(
      <Code className="language-mermaid">{PENDING_SOURCE}</Code>,
    );

    // Assert
    expect(screen.getByTestId("mermaid-diagram-pending")).toBeInTheDocument();
    expect(container).not.toHaveTextContent("P --> Q");
    await waitFor(() => expect(mermaidMock.render).toHaveBeenCalled());
    finishRender({ svg: '<svg data-testid="rendered-svg"></svg>' });
    expect(await screen.findByTestId("mermaid-diagram")).toBeInTheDocument();
    expect(
      screen.queryByTestId("mermaid-diagram-pending"),
    ).not.toBeInTheDocument();
  });

  it("shows a diagram it already rendered at once, without Mermaid", async () => {
    // Arrange
    const first = render(
      <Code className="language-mermaid">{CACHED_SOURCE}</Code>,
    );
    await screen.findByTestId("mermaid-diagram");
    first.unmount();
    mermaidMock.render.mockClear();

    // Act
    render(<Code className="language-mermaid">{CACHED_SOURCE}</Code>);

    // Assert
    expect(screen.getByTestId("mermaid-diagram")).toContainElement(
      screen.getByTestId("rendered-svg"),
    );
    expect(mermaidMock.render).not.toHaveBeenCalled();
  });

  it("leaves other languages to the syntax highlighter", () => {
    render(<Code className="language-python">{PYTHON_SOURCE}</Code>);

    expect(mermaidMock.parse).not.toHaveBeenCalled();
    expect(screen.queryByTestId("mermaid-diagram")).not.toBeInTheDocument();
  });
});
