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

const SOURCE = "flowchart LR\n  A --> B";
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
        content={`Intro text\n\n\`\`\`mermaid\n${SOURCE}\n\`\`\``}
      />,
    );

    expect(screen.getByText("Intro text")).toBeInTheDocument();
    expect(await screen.findByTestId("mermaid-diagram")).toBeInTheDocument();
  });

  it("leaves other languages to the syntax highlighter", () => {
    render(<Code className="language-python">{PYTHON_SOURCE}</Code>);

    expect(mermaidMock.parse).not.toHaveBeenCalled();
    expect(screen.queryByTestId("mermaid-diagram")).not.toBeInTheDocument();
  });
});
