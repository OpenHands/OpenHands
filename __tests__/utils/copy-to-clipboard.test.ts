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

  it("copies text via the copy event listener even if activeElement is not the textarea", async () => {
    setClipboard(undefined);
    let capturedText: string | undefined;

    document.execCommand = vi.fn().mockImplementation(() => {
      const event = new Event("copy", { bubbles: true, cancelable: true });
      const clipboardData = {
        setData: vi.fn((format: string, data: string) => {
          if (format === "text/plain") {
            capturedText = data;
          }
        }),
      };
      Object.defineProperty(event, "clipboardData", { value: clipboardData });
      document.dispatchEvent(event);
      return true;
    });

    const otherButton = document.createElement("button");
    document.body.appendChild(otherButton);
    otherButton.focus();
    expect(document.activeElement).toBe(otherButton);

    const result = await copyToClipboard("focus-independent-text");
    expect(result).toBe(true);
    expect(capturedText).toBe("focus-independent-text");
  });

  it("removes the copy event listener after a copy attempt (success or failure)", async () => {
    setClipboard(undefined);
    const removeEventListenerSpy = vi.spyOn(document, "removeEventListener");
    document.execCommand = vi.fn().mockReturnValue(true);

    await copyToClipboard("test");
    expect(removeEventListenerSpy).toHaveBeenCalledWith(
      "copy",
      expect.any(Function),
    );

    document.execCommand = vi.fn().mockImplementation(() => {
      throw new Error("execCommand failed");
    });
    await copyToClipboard("test-fail");
    expect(removeEventListenerSpy).toHaveBeenCalledTimes(2);
  });

  it("calls setSelectionRange(0, text.length) on the temporary textarea for iOS Safari", async () => {
    setClipboard(undefined);
    document.execCommand = vi.fn().mockReturnValue(true);

    let selectionRangeCalledWith: [number, number] | undefined;
    const originalSetSelectionRange =
      HTMLTextAreaElement.prototype.setSelectionRange;
    HTMLTextAreaElement.prototype.setSelectionRange = vi.fn(function (
      this: HTMLTextAreaElement,
      start: number,
      end: number,
    ) {
      selectionRangeCalledWith = [start, end];
    });

    try {
      await copyToClipboard("safari-test");
      expect(selectionRangeCalledWith).toEqual([0, "safari-test".length]);
    } finally {
      HTMLTextAreaElement.prototype.setSelectionRange =
        originalSetSelectionRange;
    }
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
    HTMLTextAreaElement.prototype.select = function select(
      this: HTMLTextAreaElement,
    ) {
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
