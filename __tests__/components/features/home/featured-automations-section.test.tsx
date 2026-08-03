import { beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import AutomationService from "#/api/automation-service/automation-service.api";
import { PinnedAutomationsDashboard } from "#/components/features/home/featured-automations/pinned-automations-dashboard";
import { RunningAutomationsList } from "#/components/features/home/featured-automations/running-automations-list";
import { NavigationProvider } from "#/context/navigation-context";
import { HOME_PINNED_AUTOMATIONS_KEY } from "#/hooks/use-home-pinned-automations";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
} from "#/types/automation";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { name?: string; count?: number }) => {
      if (options?.name) return `${key}:${options.name}`;
      if (options?.count != null) return `${key}:${options.count}`;
      return key;
    },
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

function makeAutomation(overrides: Partial<Automation> = {}): Automation {
  return {
    id: "auto-1",
    name: "Daily digest",
    prompt: "Summarize yesterday's PRs",
    trigger: {
      type: "cron",
      schedule: "0 9 * * *",
      schedule_human: "Daily at 09:00",
    },
    enabled: true,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

function makeRun(overrides: Partial<AutomationRun> = {}): AutomationRun {
  return {
    id: "run-1",
    status: AutomationRunStatus.COMPLETED,
    conversation_id: "conv-1",
    bash_command_id: "cmd-1",
    error_detail: null,
    started_at: "2026-08-01T10:00:00Z",
    completed_at: "2026-08-01T10:05:00Z",
    ...overrides,
  };
}

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
          navigate: vi.fn(),
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
  vi.mocked(AutomationService.getAutomations).mockResolvedValue({
    automations: [makeAutomation()],
    total: 1,
  });
  vi.mocked(AutomationService.getAutomationRuns).mockResolvedValue({
    runs: [makeRun()],
    total: 1,
  });
});

describe("home automations composer layout", () => {
  it("renders nothing when the automation service is unavailable", async () => {
    vi.mocked(AutomationService.checkHealth).mockResolvedValue({
      status: "error",
      message: "unreachable",
    });

    renderHomeAutomations(<RunningAutomationsList />);

    await waitFor(() =>
      expect(AutomationService.checkHealth).toHaveBeenCalledTimes(1),
    );
    expect(
      screen.queryByTestId("running-automations-list"),
    ).not.toBeInTheDocument();
    expect(AutomationService.getAutomations).not.toHaveBeenCalled();
  });

  it("renders nothing when there are no enabled automations", async () => {
    vi.mocked(AutomationService.getAutomations).mockResolvedValue({
      automations: [makeAutomation({ enabled: false })],
      total: 1,
    });

    renderHomeAutomations(<RunningAutomationsList />);

    await waitFor(() =>
      expect(AutomationService.getAutomations).toHaveBeenCalledTimes(1),
    );
    expect(
      screen.queryByTestId("running-automations-list"),
    ).not.toBeInTheDocument();
  });

  it("lists enabled automations with live run status and conversation links", async () => {
    vi.mocked(AutomationService.getAutomations).mockResolvedValue({
      automations: [
        makeAutomation({ id: "auto-1", name: "Daily digest" }),
        makeAutomation({
          id: "auto-2",
          name: "PR review",
          trigger: {
            type: "event",
            source: "github",
            on: "pull_request.opened",
          },
        }),
        makeAutomation({
          id: "auto-3",
          name: "Disabled sweep",
          enabled: false,
        }),
      ],
      total: 3,
    });
    vi.mocked(AutomationService.getAutomationRuns).mockImplementation(
      async (id: string) => {
        if (id === "auto-2") {
          return {
            runs: [
              makeRun({
                id: "run-2",
                status: AutomationRunStatus.PENDING,
                conversation_id: null,
                started_at: "1970-01-01T00:00:00Z",
                completed_at: null,
              }),
            ],
            total: 1,
          };
        }
        return { runs: [makeRun()], total: 1 };
      },
    );

    renderHomeAutomations(<RunningAutomationsList />);

    expect(
      await screen.findByTestId("running-automations-list"),
    ).toBeInTheDocument();
    expect(screen.getByText("Daily digest")).toBeInTheDocument();
    expect(screen.getByText("PR review")).toBeInTheDocument();
    expect(screen.queryByText("Disabled sweep")).not.toBeInTheDocument();

    expect(screen.getByRole("link", { name: "Daily digest" })).toHaveAttribute(
      "href",
      "/conversations/conv-1",
    );
    expect(screen.getByRole("link", { name: "PR review" })).toHaveAttribute(
      "href",
      "/automations/auto-2",
    );
    expect(screen.getByText("Daily at 09:00", { exact: false })).toBeInTheDocument();
  });

  it("pins a live automation from the row menu into the dashboard grid", async () => {
    const user = userEvent.setup();

    renderHomeAutomations(
      <>
        <PinnedAutomationsDashboard />
        <RunningAutomationsList />
      </>,
    );

    expect(
      await screen.findByTestId("running-automations-list"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("pinned-automations-dashboard"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByTestId("running-automation-menu-auto-1"));
    await user.click(screen.getByTestId("running-automation-pin-auto-1"));

    const dashboard = screen.getByTestId("pinned-automations-dashboard");
    expect(dashboard).toBeInTheDocument();
    expect(
      within(dashboard).getByTestId("pinned-automation-card-auto-1"),
    ).toBeInTheDocument();
    expect(window.localStorage.getItem(HOME_PINNED_AUTOMATIONS_KEY)).toContain(
      "auto-1",
    );

    await user.click(screen.getByTestId("unpin-automation-auto-1"));
    expect(
      screen.queryByTestId("pinned-automations-dashboard"),
    ).not.toBeInTheDocument();
  });
});
