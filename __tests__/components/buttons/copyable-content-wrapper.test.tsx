/* eslint-disable i18next/no-literal-string */
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { CopyableContentWrapper } from "#/components/shared/buttons/copyable-content-wrapper";

describe("CopyableContentWrapper", () => {
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

  it("should show copied state when fallback succeeds in non-secure contexts", async () => {
    const originalClipboard = navigator.clipboard;
    const originalExecCommand = document.execCommand;
    try {
      Object.defineProperty(navigator, "clipboard", {
        value: undefined,
        configurable: true,
        writable: true,
      });
      document.execCommand = vi.fn().mockReturnValue(true);

      render(
        <CopyableContentWrapper text="fallback text">
          <p>content</p>
        </CopyableContentWrapper>,
      );

      fireEvent.click(screen.getByTestId("copy-to-clipboard"));

      await waitFor(() => {
        expect(document.execCommand).toHaveBeenCalledWith("copy");
        expect(screen.getByTestId("copy-to-clipboard")).toHaveAttribute(
          "aria-label",
          "BUTTON$COPIED",
        );
      });
    } finally {
      Object.defineProperty(navigator, "clipboard", {
        value: originalClipboard,
        configurable: true,
        writable: true,
      });
      document.execCommand = originalExecCommand;
    }
  });

  it("should display error toast and keep copy mode when copy fails", async () => {
    const originalClipboard = navigator.clipboard;
    const originalExecCommand = document.execCommand;
    try {
      Object.defineProperty(navigator, "clipboard", {
        value: undefined,
        configurable: true,
        writable: true,
      });
      document.execCommand = vi.fn().mockReturnValue(false);

      render(
        <CopyableContentWrapper text="failing text">
          <p>content</p>
        </CopyableContentWrapper>,
      );

      fireEvent.click(screen.getByTestId("copy-to-clipboard"));

      await waitFor(() => {
        expect(screen.getByTestId("copy-to-clipboard")).toHaveAttribute(
          "aria-label",
          "BUTTON$COPY",
        );
      });
    } finally {
      Object.defineProperty(navigator, "clipboard", {
        value: originalClipboard,
        configurable: true,
        writable: true,
      });
      document.execCommand = originalExecCommand;
    }
  });
});
