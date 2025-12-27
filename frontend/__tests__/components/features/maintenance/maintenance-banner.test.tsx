import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { MemoryRouter } from "react-router";
import { MaintenanceBanner } from "#/components/features/maintenance/maintenance-banner";

// Mock react-i18next
vi.mock("react-i18next", async () => {
  const actual =
    await vi.importActual<typeof import("react-i18next")>("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string, options?: { time?: string }) => {
        const translations: Record<string, string> = {
          MAINTENANCE$SCHEDULED_MESSAGE: `Scheduled maintenance will begin at ${options?.time || "{{time}}"}`,
        };
        return translations[key] || key;
      },
    }),
  };
});

describe("MaintenanceBanner", () => {
  afterEach(() => {
    localStorage.clear();
  });

  it("renders maintenance banner with formatted time", () => {
    const startTime = "2024-01-15T10:00:00-05:00"; // EST timestamp

    const { container } = render(
      <MemoryRouter>
        <MaintenanceBanner startTime={startTime} />
      </MemoryRouter>,
    );

    // Check if the banner is rendered
    const banner = screen.queryByTestId("maintenance-banner");
    expect(banner).toBeInTheDocument();

    // Check if the warning icon (SVG) is present
    const svgIcon = container.querySelector("svg");
    expect(svgIcon).toBeInTheDocument();

    // Check if the button to close is present
    const button = within(banner!).queryByTestId("dismiss-button");
    expect(button).toBeInTheDocument();
  });

  it("handles invalid date gracefully", () => {
    // Suppress expected console.warn for invalid date parsing
    const consoleWarnSpy = vi
      .spyOn(console, "warn")
      .mockImplementation(() => {});

    const invalidTime = "invalid-date";

    render(
      <MemoryRouter>
        <MaintenanceBanner startTime={invalidTime} />
      </MemoryRouter>,
    );

    // Check if the banner is rendered
    const banner = screen.queryByTestId("maintenance-banner");
    expect(banner).not.toBeInTheDocument();

    // Restore console.warn
    consoleWarnSpy.mockRestore();
  });

  it("click on dismiss button removes banner", () => {
    const startTime = "2024-01-15T10:00:00-05:00"; // EST timestamp

    render(
      <MemoryRouter>
        <MaintenanceBanner startTime={startTime} />
      </MemoryRouter>,
    );

    // Check if the banner is rendered
    const banner = screen.queryByTestId("maintenance-banner");

    const button = within(banner!).queryByTestId("dismiss-button");
    act(() => {
      fireEvent.click(button!);
    });

    expect(banner).not.toBeInTheDocument();
  });
  it("banner reappears after dismissing on next maintenance event(future time)", () => {
    const startTime = "2024-01-15T10:00:00-05:00"; // EST timestamp
    const nextStartTime = "2025-01-15T10:00:00-05:00"; // EST timestamp

    const { rerender } = render(
      <MemoryRouter>
        <MaintenanceBanner startTime={startTime} />
      </MemoryRouter>,
    );

    // Check if the banner is rendered
    const banner = screen.queryByTestId("maintenance-banner");
    const button = within(banner!).queryByTestId("dismiss-button");

    act(() => {
      fireEvent.click(button!);
    });

    expect(banner).not.toBeInTheDocument();
    rerender(
      <MemoryRouter>
        <MaintenanceBanner startTime={nextStartTime} />
      </MemoryRouter>,
    );

    expect(screen.queryByTestId("maintenance-banner")).toBeInTheDocument();
  });
});
