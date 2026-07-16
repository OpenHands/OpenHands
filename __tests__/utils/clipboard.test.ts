import { afterEach, beforeEach, describe, it, expect, vi } from "vitest";
import { copyToClipboard } from "#/utils/clipboard";

// jsdom does not define navigator.clipboard or document.execCommand, and the
// util tests do not pull in @testing-library/user-event (which polyfills the
// clipboard). Stub both globals explicitly so the helper is exercised in
// isolation.
describe("copyToClipboard", () => {
  let writeText: ReturnType<typeof vi.fn>;
  let execCommand: ReturnType<typeof vi.fn>;
  let restoreClipboard: () => void;

  beforeEach(() => {
    writeText = vi.fn();
    execCommand = vi.fn();
    const previous = navigator.clipboard;
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    restoreClipboard = () => {
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: previous,
      });
    };
    // jsdom does not define document.execCommand, so define it directly rather
    // than spying on a missing property.
    document.execCommand =
      execCommand as unknown as typeof document.execCommand;
  });

  afterEach(() => {
    vi.restoreAllMocks();
    restoreClipboard();
    // @ts-expect-error restore jsdom's undefined
    delete document.execCommand;
  });

  it("uses navigator.clipboard.writeText when it succeeds", async () => {
    writeText.mockResolvedValue(undefined);

    const ok = await copyToClipboard("hello");

    expect(ok).toBe(true);
    expect(writeText).toHaveBeenCalledWith("hello");
    expect(execCommand).not.toHaveBeenCalled();
  });

  it("falls back to execCommand when writeText rejects", async () => {
    writeText.mockRejectedValue(new Error("denied"));
    execCommand.mockReturnValue(true);

    const ok = await copyToClipboard("recover");

    expect(ok).toBe(true);
    expect(execCommand).toHaveBeenCalledWith("copy");
  });

  it("returns false when both clipboard and execCommand fail", async () => {
    writeText.mockRejectedValue(new Error("denied"));
    execCommand.mockReturnValue(false);

    const ok = await copyToClipboard("nope");

    expect(ok).toBe(false);
  });

  it("falls back to execCommand when navigator.clipboard is undefined", async () => {
    // Drop the clipboard entirely, as in a non-secure HTTP context.
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      get: () => undefined,
    });
    execCommand.mockReturnValue(true);

    const ok = await copyToClipboard("fallback text");

    expect(ok).toBe(true);
    expect(execCommand).toHaveBeenCalledWith("copy");
  });
});
