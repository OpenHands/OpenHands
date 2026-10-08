import { describe, expect, it } from "vitest";
import { isMermaidFilePath } from "#/utils/is-mermaid-file-path";

describe("isMermaidFilePath", () => {
  it("matches mermaid extensions", () => {
    expect(isMermaidFilePath("flow.mmd")).toBe(true);
    expect(isMermaidFilePath("/workspace/project/arch.mermaid")).toBe(true);
  });

  it("matches extensions case-insensitively", () => {
    expect(isMermaidFilePath("diagram.MMD")).toBe(true);
    expect(isMermaidFilePath("notes.Mermaid")).toBe(true);
  });

  it("uses only the last extension of dotted names", () => {
    expect(isMermaidFilePath("foo.bar.mmd")).toBe(true);
    expect(isMermaidFilePath("flow.mmd.bak")).toBe(false);
  });

  it("rejects paths without a mermaid extension", () => {
    expect(isMermaidFilePath("README.md")).toBe(false);
    expect(isMermaidFilePath("/workspace/app.ts")).toBe(false);
    expect(isMermaidFilePath("")).toBe(false);
  });
});
