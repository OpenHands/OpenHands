import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { BackendUnavailable } from "#/components/features/automations/backend-not-configured";
import { I18nKey } from "#/i18n/declaration";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const translations: Record<string, string> = {
        [I18nKey.AUTOMATIONS$BACKEND_UNAVAILABLE_TITLE]:
          "Automations Unavailable",
        [I18nKey.AUTOMATIONS$BACKEND_UNAVAILABLE_MESSAGE]:
          "The automations backend is not available right now.",
        [I18nKey.AUTOMATIONS$BACKEND_UNAVAILABLE_RETRY]: "Retry",
      };
      return translations[key] || key;
    },
  }),
}));

describe("BackendUnavailable", () => {
  it("renders the unavailable message", () => {
    const onRetry = vi.fn();
    render(<BackendUnavailable onRetry={onRetry} />);

    expect(screen.getByText("Automations Unavailable")).toBeInTheDocument();
    expect(
      screen.getByText("The automations backend is not available right now."),
    ).toBeInTheDocument();
  });

  it("displays the retry button", () => {
    const onRetry = vi.fn();
    render(<BackendUnavailable onRetry={onRetry} />);

    const retryButton = screen.getByRole("button", { name: "Retry" });
    expect(retryButton).toBeInTheDocument();
  });

  it("calls onRetry when retry button is clicked", () => {
    const onRetry = vi.fn();
    render(<BackendUnavailable onRetry={onRetry} />);

    const retryButton = screen.getByRole("button", { name: "Retry" });
    fireEvent.click(retryButton);

    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
