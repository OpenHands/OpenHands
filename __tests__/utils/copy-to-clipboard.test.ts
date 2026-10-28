import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { copyToClipboard } from "#/utils/copy-to-clipboard";

describe("copyToClipboard", () => {
  const originalClipboard = navigator.clipboard;
  const originalExecCommand = document.execCommand;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    Object.defineProperty(navigator, "clipboard", {
      value: originalClipboard,
      configurable: true,
      writable: true,
    });
    document.execCommand = originalExecCommand;
  });

  it("should copy using navigator.clipboard.writeText when available", async () => {
    const writeTextMock = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: writeTextMock },
      configurable: true,
      writable: true,
    });

    const result = await copyToClipboard("hello world");

    expect(result).toBe(true);
    expect(writeTextMock).toHaveBeenCalledWith("hello world");
  });

  it("should fall back to document.execCommand when navigator.clipboard is undefined", async () => {
    Object.defineProperty(navigator, "clipboard", {
      value: undefined,
      configurable: true,
      writable: true,
    });

    const execCommandMock = vi.fn().mockReturnValue(true);
    document.execCommand = execCommandMock;

    const result = await copyToClipboard("fallback content");

    expect(result).toBe(true);
    expect(execCommandMock).toHaveBeenCalledWith("copy");
  });

  it("should fall back to document.execCommand when writeText rejects", async () => {
    const writeTextMock = vi
      .fn()
      .mockRejectedValue(new Error("Permission denied"));
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: writeTextMock },
      configurable: true,
      writable: true,
    });

    const execCommandMock = vi.fn().mockReturnValue(true);
    document.execCommand = execCommandMock;

    const result = await copyToClipboard("permission fallback");

    expect(result).toBe(true);
    expect(writeTextMock).toHaveBeenCalledWith("permission fallback");
    expect(execCommandMock).toHaveBeenCalledWith("copy");
  });

  it("should return false when both clipboard API and execCommand fail", async () => {
    const writeTextMock = vi
      .fn()
      .mockRejectedValue(new Error("NotAllowedError"));
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: writeTextMock },
      configurable: true,
      writable: true,
    });

    const execCommandMock = vi.fn().mockReturnValue(false);
    document.execCommand = execCommandMock;

    const result = await copyToClipboard("failed copy");

    expect(result).toBe(false);
  });

  it("should clean up textarea and restore selection even when execCommand throws", async () => {
    Object.defineProperty(navigator, "clipboard", {
      value: undefined,
      configurable: true,
      writable: true,
    });

    document.execCommand = vi.fn().mockImplementation(() => {
      throw new Error("execCommand error");
    });

    const removeChildSpy = vi.spyOn(document.body, "removeChild");

    const result = await copyToClipboard("throwing copy with cleanup");

    expect(result).toBe(false);
    expect(removeChildSpy).toHaveBeenCalled();
    expect(document.querySelector("textarea")).toBeNull();
  });
});
