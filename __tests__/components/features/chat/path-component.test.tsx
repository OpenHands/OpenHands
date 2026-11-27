import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  isLikelyDirectory,
  PathComponent,
} from "#/components/features/chat/path-component";

const openWorkspaceFile = vi.fn();

vi.mock("#/services/canvas-ui", () => ({
  openWorkspaceFile: (...args: unknown[]) => openWorkspaceFile(...args),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "conv-1" }),
}));

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

describe("PathComponent", () => {
  beforeEach(() => {
    openWorkspaceFile.mockClear();
  });

  it("opens the workspace file when the path is clicked", async () => {
    const user = userEvent.setup();
    const path = "docs/test.md";
    render(<PathComponent>{path}</PathComponent>);

    await user.click(screen.getByTestId("path-component-link"));

    expect(openWorkspaceFile).toHaveBeenCalledWith(path, "conv-1");
  });
});
