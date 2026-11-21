import { renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useCopyToClipboard } from "#/hooks/use-copy-to-clipboard";
import { I18nKey } from "#/i18n/declaration";
import * as toastHandlers from "#/utils/custom-toast-handlers";

const originalClipboard = navigator.clipboard;

const setClipboard = (value: unknown) =>
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    writable: true,
    value,
  });

describe("useCopyToClipboard", () => {
  afterEach(() => {
    setClipboard(originalClipboard);
    vi.restoreAllMocks();
  });

  it("resolves true and shows no toast when the copy works", async () => {
    const displayErrorToast = vi
      .spyOn(toastHandlers, "displayErrorToast")
      .mockImplementation(() => {});
    setClipboard({ writeText: vi.fn().mockResolvedValue(undefined) });
    const { result } = renderHook(() => useCopyToClipboard());

    await expect(result.current("hello")).resolves.toBe(true);
    expect(displayErrorToast).not.toHaveBeenCalled();
  });

  it("resolves false and shows the generic toast when the copy fails", async () => {
    const displayErrorToast = vi
      .spyOn(toastHandlers, "displayErrorToast")
      .mockImplementation(() => {});
    setClipboard(undefined);
    document.execCommand = vi.fn().mockReturnValue(false);
    const { result } = renderHook(() => useCopyToClipboard());

    await expect(result.current("hello")).resolves.toBe(false);
    expect(displayErrorToast).toHaveBeenCalledWith(I18nKey.COMMON$COPY_FAILED);
  });
});
