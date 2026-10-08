import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

const mermaidMock = vi.hoisted(() => ({
  initialize: vi.fn(),
  parse: vi.fn(async () => ({})),
  render: vi.fn(async () => ({ svg: "<svg><text>diagram body</text></svg>" })),
}));

vi.mock("mermaid", () => ({ default: mermaidMock }));

import { MermaidDiagram } from "#/components/features/markdown/mermaid-diagram";
import { resetMermaidLoaderForTests } from "#/components/features/markdown/mermaid-renderer";

const SOURCE = "graph TD;\n  A-->B;";

describe("MermaidDiagram", () => {
  beforeEach(() => {
    resetMermaidLoaderForTests();
    mermaidMock.initialize.mockClear();
    mermaidMock.parse.mockClear();
    mermaidMock.render.mockReset();
    mermaidMock.render.mockResolvedValue({
      svg: "<svg><text>diagram body</text></svg>",
    });
  });

  it("renders the sanitized SVG returned by the renderer", async () => {
    render(<MermaidDiagram source={SOURCE} />);

    await waitFor(() =>
      expect(screen.getByTestId("mermaid-diagram-svg")).toBeInTheDocument(),
    );
    expect(screen.getByText("diagram body")).toBeInTheDocument();
  });

  it("renders the exact fenced-block source", async () => {
    render(<MermaidDiagram source={SOURCE} />);

    await waitFor(() =>
      expect(mermaidMock.render).toHaveBeenCalledWith(
        expect.any(String),
        SOURCE,
      ),
    );
  });

  it("fails soft: shows the source and a message when rendering rejects", async () => {
    mermaidMock.parse.mockRejectedValueOnce(new Error("bad diagram"));

    render(<MermaidDiagram source={SOURCE} />);

    await waitFor(() =>
      expect(screen.getByTestId("mermaid-diagram-error")).toBeInTheDocument(),
    );
    // The raw source remains visible so the diagram is never lost.
    expect(screen.getByTestId("mermaid-diagram-source")).toHaveTextContent(
      "graph TD;",
    );
  });

  it("toggles to the raw source and back", async () => {
    const user = userEvent.setup();
    render(<MermaidDiagram source={SOURCE} />);

    await waitFor(() =>
      expect(screen.getByTestId("mermaid-diagram-svg")).toBeInTheDocument(),
    );

    await user.click(screen.getByTestId("mermaid-diagram-toggle-source"));
    expect(screen.getByTestId("mermaid-diagram-source")).toHaveTextContent(
      "graph TD;",
    );

    await user.click(screen.getByTestId("mermaid-diagram-toggle-source"));
    expect(
      screen.queryByTestId("mermaid-diagram-source"),
    ).not.toBeInTheDocument();
  });

  it("copies the raw source to the clipboard", async () => {
    const user = userEvent.setup();
    render(<MermaidDiagram source={SOURCE} />);

    await waitFor(() =>
      expect(screen.getByTestId("mermaid-diagram-svg")).toBeInTheDocument(),
    );
    await user.click(screen.getByTestId("mermaid-diagram-copy-source"));

    await waitFor(() =>
      expect(navigator.clipboard.readText()).resolves.toBe(SOURCE),
    );
  });

  it("names the downloaded SVG after the artifact file", async () => {
    const user = userEvent.setup();
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});
    render(<MermaidDiagram source={SOURCE} fileName="flow.mmd" />);

    await waitFor(() =>
      expect(screen.getByTestId("mermaid-diagram-svg")).toBeInTheDocument(),
    );

    let downloadName = "";
    const appendSpy = vi
      .spyOn(document.body, "appendChild")
      .mockImplementation((node) => {
        downloadName = (node as HTMLAnchorElement).download;
        return node;
      });

    await user.click(screen.getByTestId("mermaid-diagram-download-svg"));

    expect(downloadName).toBe("flow.svg");
    appendSpy.mockRestore();
    clickSpy.mockRestore();
  });

  it("exposes a View action only when a file deep link is provided", async () => {
    const { rerender } = render(<MermaidDiagram source={SOURCE} />);

    await waitFor(() =>
      expect(screen.getByTestId("mermaid-diagram-svg")).toBeInTheDocument(),
    );
    // A fenced block has no backing file, so the affordance is hidden.
    expect(
      screen.queryByTestId("mermaid-diagram-view"),
    ).not.toBeInTheDocument();

    const onView = vi.fn();
    rerender(<MermaidDiagram source={SOURCE} onView={onView} />);

    await waitFor(() =>
      expect(screen.getByTestId("mermaid-diagram-view")).toBeInTheDocument(),
    );
    screen.getByTestId("mermaid-diagram-view").click();
    expect(onView).toHaveBeenCalledTimes(1);
  });

  it("shows the source instead of Loading when toggled during a slow render", async () => {
    const user = userEvent.setup();
    // Never resolves: keeps the card in the loading state for the assertion.
    mermaidMock.render.mockReturnValue(new Promise(() => {}));

    render(<MermaidDiagram source={SOURCE} />);

    await user.click(screen.getByTestId("mermaid-diagram-toggle-source"));

    expect(screen.getByTestId("mermaid-diagram-source")).toHaveTextContent(
      "graph TD;",
    );
    expect(screen.queryByText("Loading")).not.toBeInTheDocument();
  });
});
