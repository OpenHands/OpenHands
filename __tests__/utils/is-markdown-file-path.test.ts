import { describe, expect, it } from "vitest";
import { isMarkdownFilePath } from "#/utils/is-markdown-file-path";

describe("isMarkdownFilePath", () => {
  it("matches markdown extensions", () => {
    expect(isMarkdownFilePath("canvas.md")).toBe(true);
    expect(isMarkdownFilePath("/workspace/project/report.markdown")).toBe(true);
    expect(isMarkdownFilePath("docs/page.mdx")).toBe(true);
  });

  it("matches extensions case-insensitively", () => {
    expect(isMarkdownFilePath("README.MD")).toBe(true);
    expect(isMarkdownFilePath("notes.Markdown")).toBe(true);
  });

  it("uses only the last extension of dotted names", () => {
    expect(isMarkdownFilePath("foo.bar.md")).toBe(true);
    expect(isMarkdownFilePath("report.md.bak")).toBe(false);
  });

  it("rejects paths without a markdown extension", () => {
    expect(isMarkdownFilePath("Makefile")).toBe(false);
    expect(isMarkdownFilePath("/workspace/app.ts")).toBe(false);
    expect(isMarkdownFilePath("")).toBe(false);
  });
});
