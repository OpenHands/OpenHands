import { beforeEach, describe, expect, it, vi } from "vitest";

const mermaidMock = vi.hoisted(() => ({
  initialize: vi.fn(),
  parse: vi.fn(async () => ({})),
  render: vi.fn(),
}));

vi.mock("mermaid", () => ({ default: mermaidMock }));

import {
  MERMAID_SECURITY_LEVEL,
  renderMermaidToSafeSvg,
  resetMermaidLoaderForTests,
  sanitizeMermaidSvg,
} from "#/components/features/markdown/mermaid-renderer";

const MALICIOUS_SVG = [
  '<svg xmlns="http://www.w3.org/2000/svg">',
  "<script>window.__pwned = true;</script>",
  '<a href="javascript:alert(1)" onload="alert(2)">link</a>',
  '<image href="javascript:alert(3)" />',
  "<foreignObject><body></body></foreignObject>",
  "<text>keep me</text>",
  "</svg>",
].join("");

describe("sanitizeMermaidSvg", () => {
  it("strips scripts, event handlers, javascript: URLs and foreignObject", () => {
    const safe = sanitizeMermaidSvg(MALICIOUS_SVG);

    expect(safe).not.toContain("<script");
    expect(safe).not.toContain("onload");
    expect(safe.toLowerCase()).not.toContain("javascript:");
    expect(safe.toLowerCase()).not.toContain("foreignobject");
    // Legitimate diagram content survives.
    expect(safe).toContain("keep me");
  });
});

describe("renderMermaidToSafeSvg", () => {
  beforeEach(() => {
    resetMermaidLoaderForTests();
    mermaidMock.initialize.mockClear();
    mermaidMock.parse.mockClear();
    mermaidMock.render.mockReset();
  });

  it("does not load or configure mermaid until a diagram is rendered", async () => {
    // The dynamic-import boundary: importing this module must not pull in the
    // (heavy) mermaid library. Only an actual render triggers initialization.
    expect(mermaidMock.initialize).not.toHaveBeenCalled();
    expect(mermaidMock.render).not.toHaveBeenCalled();
  });

  it("initializes mermaid in strict mode with HTML labels disabled", async () => {
    mermaidMock.render.mockResolvedValue({ svg: "<svg><text>ok</text></svg>" });

    await renderMermaidToSafeSvg("d1", "graph TD; A-->B;");

    expect(mermaidMock.initialize).toHaveBeenCalledWith(
      expect.objectContaining({
        startOnLoad: false,
        securityLevel: MERMAID_SECURITY_LEVEL,
        htmlLabels: false,
      }),
    );
    expect(MERMAID_SECURITY_LEVEL).toBe("strict");
  });

  it("returns sanitized SVG even when mermaid emits unsafe markup", async () => {
    mermaidMock.render.mockResolvedValue({ svg: MALICIOUS_SVG });

    const svg = await renderMermaidToSafeSvg("d2", "graph TD; A-->B;");

    expect(svg).not.toContain("<script");
    expect(svg.toLowerCase()).not.toContain("javascript:");
    expect(svg).toContain("keep me");
  });

  it("parses before rendering so invalid diagrams reject", async () => {
    mermaidMock.parse.mockRejectedValueOnce(new Error("bad diagram"));

    await expect(renderMermaidToSafeSvg("d3", "not a diagram")).rejects.toThrow(
      "bad diagram",
    );
    expect(mermaidMock.render).not.toHaveBeenCalled();
  });
});
