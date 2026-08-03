import { beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import AutomationService from "#/api/automation-service/automation-service.api";
import {
  AutomationsToaster,
  HOME_AUTOMATIONS_TOASTER_DISMISSED_KEY,
} from "#/components/features/home/featured-automations/automations-toaster";
import { HOME_AUTOMATION_ACTIVITY_EXAMPLES } from "#/components/features/home/featured-automations/home-automation-activity-examples";
import { HOME_RECOMMENDED_AUTOMATION_CARDS } from "#/components/features/home/featured-automations/home-recommended-automation-examples";
import { PinnedAutomationsDashboard } from "#/components/features/home/featured-automations/pinned-automations-dashboard";
import { RecommendedAutomationsRail } from "#/components/features/home/featured-automations/recommended-automations-rail";
import { RunningAutomationsList } from "#/components/features/home/featured-automations/running-automations-list";
import { NavigationProvider } from "#/context/navigation-context";
import { HOME_PINNED_AUTOMATIONS_KEY } from "#/hooks/use-home-pinned-automations";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { name?: string }) =>
      options?.name ? `${key}:${options.name}` : key,
    i18n: { language: "en" },
  }),
}));

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    checkHealth: vi.fn(),
    getAutomations: vi.fn(),
    getAutomationRuns: vi.fn(),
  },
}));

const navigate = vi.fn();

function renderHomeAutomations(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <NavigationProvider
        value={{
          currentPath: "/",
          conversationId: null,
          isNavigating: false,
          navigate,
        }}
      >
        {ui}
      </NavigationProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  window.localStorage.clear();
  vi.mocked(AutomationService.checkHealth).mockResolvedValue({ status: "ok" });
});

describe("home automations composer layout", () => {
  it("shows a dismissible automations toaster with a Start action", async () => {
    const user = userEvent.setup();
    renderHomeAutomations(<AutomationsToaster />);

    expect(
      await screen.findByTestId("home-automations-toaster"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("home-automations-toaster-start"));
    expect(navigate).toHaveBeenCalledWith("/automations");

    await user.click(screen.getByTestId("home-automations-toaster-dismiss"));
    expect(
      screen.queryByTestId("home-automations-toaster"),
    ).not.toBeInTheDocument();
    expect(
      window.localStorage.getItem(HOME_AUTOMATIONS_TOASTER_DISMISSED_KEY),
    ).toBe("true");
  });

  it("hides the toaster when previously dismissed", async () => {
    window.localStorage.setItem(
      HOME_AUTOMATIONS_TOASTER_DISMISSED_KEY,
      "true",
    );
    renderHomeAutomations(<AutomationsToaster />);

    await waitFor(() =>
      expect(AutomationService.checkHealth).toHaveBeenCalled(),
    );
    expect(
      screen.queryByTestId("home-automations-toaster"),
    ).not.toBeInTheDocument();
  });

  it("renders compact recommended automation starter cards", () => {
    renderHomeAutomations(<RecommendedAutomationsRail />);

    expect(
      screen.getByTestId("recommended-automations-rail"),
    ).toBeInTheDocument();

    for (const card of HOME_RECOMMENDED_AUTOMATION_CARDS) {
      const link = screen.getByTestId(`recommended-automation-card-${card.id}`);
      expect(link).toHaveAttribute("href", card.href);
      expect(link).toHaveTextContent(card.labelKey);
      expect(link.className).toContain("min-h-[4.25rem]");
    }
  });

  it("renders static example rows for running and recent activity", () => {
    renderHomeAutomations(<RunningAutomationsList />);

    expect(screen.getByTestId("running-automations-list")).toBeInTheDocument();
    expect(AutomationService.getAutomations).not.toHaveBeenCalled();

    for (const example of HOME_AUTOMATION_ACTIVITY_EXAMPLES) {
      expect(screen.getByText(example.name)).toBeInTheDocument();
      expect(
        screen.getByText(example.whenLabel, { exact: false }),
      ).toBeInTheDocument();
    }

    const first = HOME_AUTOMATION_ACTIVITY_EXAMPLES[0];
    expect(screen.getByRole("link", { name: first.name })).toHaveAttribute(
      "href",
      `/conversations/${first.conversationId}`,
    );

    const pending = HOME_AUTOMATION_ACTIVITY_EXAMPLES.find(
      (example) => example.conversationId === null,
    );
    expect(pending).toBeDefined();
    expect(screen.getByRole("link", { name: pending!.name })).toHaveAttribute(
      "href",
      `/automations/${pending!.id}`,
    );
  });

  it("pins an automation from the row menu into the dashboard grid", async () => {
    const user = userEvent.setup();
    const first = HOME_AUTOMATION_ACTIVITY_EXAMPLES[0];

    renderHomeAutomations(
      <>
        <PinnedAutomationsDashboard />
        <RunningAutomationsList />
      </>,
    );

    expect(
      screen.queryByTestId("pinned-automations-dashboard"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByTestId(`running-automation-menu-${first.id}`));
    await user.click(screen.getByTestId(`running-automation-pin-${first.id}`));

    const dashboard = screen.getByTestId("pinned-automations-dashboard");
    expect(dashboard).toBeInTheDocument();
    expect(
      within(dashboard).getByTestId(`pinned-automation-card-${first.id}`),
    ).toBeInTheDocument();
    expect(window.localStorage.getItem(HOME_PINNED_AUTOMATIONS_KEY)).toContain(
      first.id,
    );

    await user.click(screen.getByTestId(`unpin-automation-${first.id}`));
    expect(
      screen.queryByTestId("pinned-automations-dashboard"),
    ).not.toBeInTheDocument();
  });
});
