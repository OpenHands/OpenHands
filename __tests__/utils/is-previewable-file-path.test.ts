import { describe, expect, it } from "vitest";
import {
  getArtifactPreviewKind,
  getFileExtension,
  isFramePreviewablePath,
  isPreviewableArtifactPath,
} from "#/utils/is-previewable-file-path";

describe("getFileExtension", () => {
  it("returns the lowercased extension", () => {
    expect(getFileExtension("index.HTML")).toBe("html");
    expect(getFileExtension("/workspace/app/page.svg")).toBe("svg");
  });

  it("returns an empty string when there is no extension", () => {
    expect(getFileExtension("Makefile")).toBe("");
    expect(getFileExtension("")).toBe("");
  });
});

describe("isFramePreviewablePath", () => {
  it("accepts HTML and SVG variants", () => {
    expect(isFramePreviewablePath("index.html")).toBe(true);
    expect(isFramePreviewablePath("/workspace/page.htm")).toBe(true);
    expect(isFramePreviewablePath("assets/icon.SVG")).toBe(true);
  });

  it("rejects non-frame types", () => {
    expect(isFramePreviewablePath("notes.md")).toBe(false);
    expect(isFramePreviewablePath("app.ts")).toBe(false);
    expect(isFramePreviewablePath("index.html.bak")).toBe(false);
    expect(isFramePreviewablePath("")).toBe(false);
  });
});

describe("getArtifactPreviewKind", () => {
  it("classifies each supported artifact format", () => {
    expect(getArtifactPreviewKind("report.md")).toBe("markdown");
    expect(getArtifactPreviewKind("index.html")).toBe("frame");
    expect(getArtifactPreviewKind("chart.svg")).toBe("frame");
  });

  it("returns null for binary and plain-source paths", () => {
    // Binary formats (raster images, PDFs) and Office documents are
    // deliberately not previewed inline (#18113); they keep their existing
    // CodeBlock / fallback rendering.
    expect(getArtifactPreviewKind("logo.png")).toBe(null);
    expect(getArtifactPreviewKind("photo.jpg")).toBe(null);
    expect(getArtifactPreviewKind("spec.pdf")).toBe(null);
    expect(getArtifactPreviewKind("report.docx")).toBe(null);
    expect(getArtifactPreviewKind("app.tsx")).toBe(null);
    expect(getArtifactPreviewKind("Makefile")).toBe(null);
    expect(getArtifactPreviewKind("archive.zip")).toBe(null);
  });
});

describe("isPreviewableArtifactPath", () => {
  it("covers every rich-preview format", () => {
    expect(isPreviewableArtifactPath("report.md")).toBe(true);
    expect(isPreviewableArtifactPath("index.html")).toBe(true);
    expect(isPreviewableArtifactPath("icon.svg")).toBe(true);
  });

  it("rejects binary formats and plain source files", () => {
    expect(isPreviewableArtifactPath("logo.png")).toBe(false);
    expect(isPreviewableArtifactPath("spec.pdf")).toBe(false);
    expect(isPreviewableArtifactPath("app.tsx")).toBe(false);
    expect(isPreviewableArtifactPath("Makefile")).toBe(false);
  });
});
