import { render, screen } from "@testing-library/react";
import { test, expect, describe, vi } from "vitest";
import { CopyToClipboardButton } from "#/components/shared/buttons/copy-to-clipboard-button";

// Mock react-i18next
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

describe("CopyToClipboardButton", () => {
  test("should have localized aria-label", () => {
    render(
      <CopyToClipboardButton
        isHidden={false}
        isDisabled={false}
        onClick={() => {}}
        mode="copy"
      />
    );

    const button = screen.getByTestId("copy-to-clipboard");
    expect(button).toHaveAttribute("aria-label", "BUTTON$COPY");
  });

  test("should have localized aria-label when copied", () => {
    render(
      <CopyToClipboardButton
        isHidden={false}
        isDisabled={false}
        onClick={() => {}}
        mode="copied"
      />
    );

    const button = screen.getByTestId("copy-to-clipboard");
    expect(button).toHaveAttribute("aria-label", "BUTTON$COPIED");
  });
});
