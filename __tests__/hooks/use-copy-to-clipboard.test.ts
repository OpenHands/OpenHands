import { describe, expect, it, vi, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useCopyToClipboard } from "#/hooks/use-copy-to-clipboard";
import * as copyUtil from "#/utils/copy-to-clipboard";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
}));

describe("useCopyToClipboard", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns true and does not display an error toast on successful copy", async () => {
    vi.spyOn(copyUtil, "copyToClipboard").mockResolvedValue(true);
    const { result } = renderHook(() => useCopyToClipboard());

    let success: boolean | undefined;
    await act(async () => {
      success = await result.current("hello world");
    });

    expect(success).toBe(true);
    expect(copyUtil.copyToClipboard).toHaveBeenCalledWith("hello world");
    expect(displayErrorToast).not.toHaveBeenCalled();
  });

  it("returns false and displays generic error toast on failed copy", async () => {
    vi.spyOn(copyUtil, "copyToClipboard").mockResolvedValue(false);
    const { result } = renderHook(() => useCopyToClipboard());

    let success: boolean | undefined;
    await act(async () => {
      success = await result.current("failing text");
    });

    expect(success).toBe(false);
    expect(copyUtil.copyToClipboard).toHaveBeenCalledWith("failing text");
    expect(displayErrorToast).toHaveBeenCalledWith(
      I18nKey.CLIPBOARD$COPY_FAILED,
    );
  });
});
