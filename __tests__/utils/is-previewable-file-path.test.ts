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
    expect(isFramePreviewablePath("logo.png")).toBe(false);
    expect(isFramePreviewablePath("index.html.bak")).toBe(false);
    expect(isFramePreviewablePath("")).toBe(false);
  });
});

describe("getArtifactPreviewKind", () => {
  it("classifies each supported artifact format", () => {
    expect(getArtifactPreviewKind("report.md")).toBe("markdown");
    expect(getArtifactPreviewKind("index.html")).toBe("frame");
    expect(getArtifactPreviewKind("chart.svg")).toBe("frame");
    expect(getArtifactPreviewKind("logo.PNG")).toBe("image");
    expect(getArtifactPreviewKind("photo.jpg")).toBe("image");
    expect(getArtifactPreviewKind("photo.jpeg")).toBe("image");
    expect(getArtifactPreviewKind("spec.pdf")).toBe("pdf");
  });

  it("returns null for paths with no rich preview", () => {
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
    expect(isPreviewableArtifactPath("logo.png")).toBe(true);
    expect(isPreviewableArtifactPath("spec.pdf")).toBe(true);
  });

  it("rejects plain source files", () => {
    expect(isPreviewableArtifactPath("app.tsx")).toBe(false);
    expect(isPreviewableArtifactPath("Makefile")).toBe(false);
  });
});
