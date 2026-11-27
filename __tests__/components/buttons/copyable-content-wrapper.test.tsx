import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, it, expect, vi } from "vitest";
import { CopyableContentWrapper } from "#/components/shared/buttons/copyable-content-wrapper";
import { I18nKey } from "#/i18n/declaration";
import * as toastHandlers from "#/utils/custom-toast-handlers";

const originalClipboard = navigator.clipboard;
const originalExecCommand = document.execCommand;

describe("CopyableContentWrapper", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      writable: true,
      value: originalClipboard,
    });
    Object.defineProperty(document, "execCommand", {
      configurable: true,
      writable: true,
      value: originalExecCommand,
    });
  });
  it("should hide the copy button by default", () => {
    render(
      <CopyableContentWrapper text="hello">
        <p>content</p>
      </CopyableContentWrapper>,
    );

    expect(screen.getByTestId("copy-to-clipboard")).not.toBeVisible();
  });

  it("should show the copy button on hover", async () => {
    const user = userEvent.setup();
    render(
      <CopyableContentWrapper text="hello">
        <p>content</p>
      </CopyableContentWrapper>,
    );

    await user.hover(screen.getByText("content"));

    expect(screen.getByTestId("copy-to-clipboard")).toBeVisible();
  });

  it("should copy text to clipboard on click", async () => {
    const user = userEvent.setup();
    render(
      <CopyableContentWrapper text="copy me">
        <p>content</p>
      </CopyableContentWrapper>,
    );

    await user.click(screen.getByTestId("copy-to-clipboard"));

    await waitFor(() =>
      expect(navigator.clipboard.readText()).resolves.toBe("copy me"),
    );
  });

  it("should show copied state after clicking", async () => {
    const user = userEvent.setup();
    render(
      <CopyableContentWrapper text="hello">
        <p>content</p>
      </CopyableContentWrapper>,
    );

    await user.click(screen.getByTestId("copy-to-clipboard"));

    expect(screen.getByTestId("copy-to-clipboard")).toHaveAttribute(
      "aria-label",
      "BUTTON$COPIED",
    );
  });

  it("should show copied state when the fallback succeeds", async () => {
    const user = userEvent.setup();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      writable: true,
      value: undefined,
    });
    document.execCommand = vi.fn().mockReturnValue(true);
    render(
      <CopyableContentWrapper text="fallback text">
        <p>content</p>
      </CopyableContentWrapper>,
    );

    await user.click(screen.getByTestId("copy-to-clipboard"));

    await waitFor(() => {
      expect(screen.getByTestId("copy-to-clipboard")).toHaveAttribute(
        "aria-label",
        "BUTTON$COPIED",
      );
    });
    expect(document.execCommand).toHaveBeenCalledWith("copy");
  });

  it("should show an error instead of copied state when every mechanism fails", async () => {
    const displayErrorToast = vi
      .spyOn(toastHandlers, "displayErrorToast")
      .mockImplementation(() => {});
    const user = userEvent.setup();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      writable: true,
      value: undefined,
    });
    document.execCommand = vi.fn().mockReturnValue(false);
    render(
      <CopyableContentWrapper text="failing text">
        <p>content</p>
      </CopyableContentWrapper>,
    );

    await user.click(screen.getByTestId("copy-to-clipboard"));

    await waitFor(() => {
      expect(displayErrorToast).toHaveBeenCalledWith(
        I18nKey.CHAT_INTERFACE$CHAT_MESSAGE_COPY_FAILED,
      );
    });
    expect(screen.getByTestId("copy-to-clipboard")).toHaveAttribute(
      "aria-label",
      "BUTTON$COPY",
    );
  });
});
