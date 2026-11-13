import { describe, expect, it, vi, afterEach } from "vitest";
import { copyToClipboard } from "#/utils/copy-to-clipboard";

describe("copyToClipboard", () => {
  const originalClipboard = navigator.clipboard;

  afterEach(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      writable: true,
      value: originalClipboard,
    });
    vi.restoreAllMocks();
  });

  const setClipboard = (value: unknown) =>
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      writable: true,
      value,
    });

  it("uses the async Clipboard API when it is available", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    setClipboard({ writeText });
    const execCommand = vi.fn();
    document.execCommand = execCommand;

    await expect(copyToClipboard("hello")).resolves.toBe(true);
    expect(writeText).toHaveBeenCalledWith("hello");
    expect(execCommand).not.toHaveBeenCalled();
  });

  it("falls back to execCommand when the Clipboard API is missing", async () => {
    // A non-secure context has no navigator.clipboard at all, which is the
    // reported symptom: the copy button did nothing.
    setClipboard(undefined);
    const execCommand = vi.fn().mockReturnValue(true);
    document.execCommand = execCommand;

    await expect(copyToClipboard("hello")).resolves.toBe(true);
    expect(execCommand).toHaveBeenCalledWith("copy");
  });

  it("falls back to execCommand when writeText rejects", async () => {
    const writeText = vi.fn().mockRejectedValue(new Error("NotAllowedError"));
    setClipboard({ writeText });
    const execCommand = vi.fn().mockReturnValue(true);
    document.execCommand = execCommand;

    await expect(copyToClipboard("hello")).resolves.toBe(true);
    expect(writeText).toHaveBeenCalledWith("hello");
    expect(execCommand).toHaveBeenCalledWith("copy");
  });

  it("reports failure when every mechanism fails", async () => {
    setClipboard({ writeText: vi.fn().mockRejectedValue(new Error("denied")) });
    document.execCommand = vi.fn().mockReturnValue(false);

    await expect(copyToClipboard("hello")).resolves.toBe(false);
  });

  it("removes the temporary textarea it appends", async () => {
    setClipboard(undefined);
    document.execCommand = vi.fn().mockReturnValue(true);
    const before = document.body.children.length;

    await copyToClipboard("hello");

    expect(document.body.children.length).toBe(before);
  });
});

describe("copyToClipboard focus restoration", () => {
  const originalClipboard = navigator.clipboard;
  const originalSelect = HTMLTextAreaElement.prototype.select;

  afterEach(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      writable: true,
      value: originalClipboard,
    });
    HTMLTextAreaElement.prototype.select = originalSelect;
    vi.restoreAllMocks();
    document.body.innerHTML = "";
  });

  it("returns focus to the element that had it before the fallback", async () => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      writable: true,
      value: undefined,
    });
    document.execCommand = vi.fn().mockReturnValue(true);

    // jsdom's select() is a no-op; a real browser moves focus onto the
    // textarea, which is what strands the keyboard user once it is removed.
    HTMLTextAreaElement.prototype.select = function select(this: HTMLTextAreaElement) {
      this.focus();
    };

    const button = document.createElement("button");
    document.body.appendChild(button);
    button.focus();
    expect(document.activeElement).toBe(button);

    await copyToClipboard("hello");

    expect(document.activeElement).toBe(button);
  });
});
