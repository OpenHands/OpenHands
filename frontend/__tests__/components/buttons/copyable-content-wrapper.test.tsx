import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
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
});
