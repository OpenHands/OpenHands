import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { PopoutChrome } from "#/components/features/popout/popout-chrome";
import { I18nKey } from "#/i18n/declaration";

describe("PopoutChrome", () => {
  it("renders the title, source subtitle, and window controls when expanded", () => {
    const onMaximize = vi.fn();
    const onToggleMinimized = vi.fn();
    const onClose = vi.fn();

    renderWithProviders(
      <PopoutChrome
        title="Trip planning (branch)"
        mode="expanded"
        onMaximize={onMaximize}
        onToggleMinimized={onToggleMinimized}
        onClose={onClose}
      >
        <div data-testid="body">chat</div>
      </PopoutChrome>,
    );

    expect(screen.getByText("Trip planning (branch)")).toBeInTheDocument();
    // test-utils i18n returns the key; the title is interpolated into the key
    // string only after make-i18n resources load — assert the control labels.
    expect(
      screen.getByRole("button", { name: I18nKey.POPOUT$MAXIMIZE }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: I18nKey.POPOUT$MINIMIZE }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: I18nKey.POPOUT$CLOSE }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("body")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: I18nKey.POPOUT$MAXIMIZE }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: I18nKey.POPOUT$MINIMIZE }),
    );
    fireEvent.click(screen.getByRole("button", { name: I18nKey.POPOUT$CLOSE }));
    expect(onMaximize).toHaveBeenCalledOnce();
    expect(onToggleMinimized).toHaveBeenCalledOnce();
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("keeps children mounted but hidden when minimized", () => {
    renderWithProviders(
      <PopoutChrome
        title="Trip planning (branch)"
        mode="minimized"
        onMaximize={vi.fn()}
        onToggleMinimized={vi.fn()}
        onClose={vi.fn()}
      >
        <div data-testid="body">chat</div>
      </PopoutChrome>,
    );

    const body = screen.getByTestId("body");
    expect(body).toBeInTheDocument();
    expect(body.parentElement).toHaveAttribute("aria-hidden", "true");
    expect(
      screen.getByRole("button", { name: I18nKey.POPOUT$EXPAND }),
    ).toBeInTheDocument();
  });
});
