import { describe, expect, it } from "vitest";
import { isLikelyDirectory } from "#/components/features/chat/path-component";

describe("isLikelyDirectory", () => {
  it("should return false for empty path", () => {
    expect(isLikelyDirectory("")).toBe(false);
  });

  it("should return true for paths ending with forward slash", () => {
    expect(isLikelyDirectory("/path/to/dir/")).toBe(true);
    expect(isLikelyDirectory("dir/")).toBe(true);
  });

  it("should return true for paths ending with backslash", () => {
    expect(isLikelyDirectory("C:\\path\\to\\dir\\")).toBe(true);
    expect(isLikelyDirectory("dir\\")).toBe(true);
  });

  it("should return true for paths without extension", () => {
    expect(isLikelyDirectory("/path/to/dir")).toBe(true);
    expect(isLikelyDirectory("dir")).toBe(true);
  });

  it("should return false for paths ending with dot", () => {
    expect(isLikelyDirectory("/path/to/dir.")).toBe(false);
    expect(isLikelyDirectory("dir.")).toBe(false);
  });

  it("should return false for paths with file extensions", () => {
    expect(isLikelyDirectory("/path/to/file.txt")).toBe(false);
    expect(isLikelyDirectory("file.js")).toBe(false);
    expect(isLikelyDirectory("script.test.ts")).toBe(false);
  });
});
