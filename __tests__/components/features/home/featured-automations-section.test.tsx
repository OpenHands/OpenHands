import { beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import AutomationService from "#/api/automation-service/automation-service.api";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { PinnedAutomationsDashboard } from "#/components/features/home/featured-automations/pinned-automations-dashboard";
import { RunningAutomationsList } from "#/components/features/home/featured-automations/running-automations-list";
import { NavigationProvider } from "#/context/navigation-context";
import { HOME_PINNED_AUTOMATIONS_KEY } from "#/hooks/use-home-pinned-automations";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
} from "#/types/automation";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

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
    dispatchAutomation: vi.fn(),
    toggleAutomation: vi.fn(),
    cancelAutomationRun: vi.fn(),
  },
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
}));

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: { batchGetAppConversations: vi.fn() },
  }),
);

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

function makeConversation(id: string, title: string | null): AppConversation {
  return {
    id,
    created_by_user_id: null,
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    title,
    trigger: null,
    pr_number: [],
    llm_model: null,
    metrics: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    execution_status: null,
    sandbox_status: null,
    conversation_url: "https://sandbox.example.com/api",
    session_api_key: null,
    sandbox_id: null,
    sub_conversation_ids: [],
  };
}

/** Pins are stored under a backend/org-scoped key; find it by prefix. */
function getStoredPinnedIds(): string | null {
  const key = Object.keys(window.localStorage).find((storageKey) =>
    storageKey.startsWith(HOME_PINNED_AUTOMATIONS_KEY),
  );
  return key ? window.localStorage.getItem(key) : null;
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
  vi.mocked(AutomationService.dispatchAutomation).mockResolvedValue(makeRun());
  vi.mocked(AutomationService.toggleAutomation).mockResolvedValue(
    makeAutomation({ enabled: false }),
  );
  vi.mocked(
    AgentServerConversationService.batchGetAppConversations,
  ).mockResolvedValue([]);
});

describe("home automations composer layout", () => {
  it("lists rows with accessible health labels and a manage control", async () => {
    vi.mocked(AutomationService.getAutomations).mockResolvedValue({
      automations: [
        makeAutomation({ id: "auto-1", name: "Daily digest" }),
        makeAutomation({ id: "auto-2", name: "PR review" }),
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
                status: AutomationRunStatus.FAILED,
                error_detail: "boom",
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
      await screen.findByRole("link", {
        name: /Daily digest\s*FEATURED_AUTOMATIONS\$LAST_RUN_SUCCEEDED/,
      }),
    ).toBeInTheDocument();
    expect(
      await screen.findByRole("link", {
        name: /PR review\s*FEATURED_AUTOMATIONS\$LAST_RUN_FAILED/,
      }),
    ).toBeInTheDocument();
    expect(screen.queryByText("Disabled sweep")).not.toBeInTheDocument();
    expect(screen.getByTestId("home-automations-manage")).toHaveAttribute(
      "href",
      "/automations",
    );
  });

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

  it("keeps the pane visible with an add prompt when there are no enabled automations", async () => {
    vi.mocked(AutomationService.getAutomations).mockResolvedValue({
      automations: [makeAutomation({ enabled: false })],
      total: 1,
    });

    renderHomeAutomations(<RunningAutomationsList />);

    await waitFor(() =>
      expect(AutomationService.getAutomations).toHaveBeenCalledTimes(1),
    );
    // The pane stays visible to nudge users toward adding automations.
    expect(
      await screen.findByTestId("running-automations-list"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("running-automations-empty-hint"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("home-automations-manage")).toHaveAttribute(
      "href",
      "/automations",
    );
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
      ],
      total: 2,
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

    expect(screen.getByRole("link", { name: /Daily digest/ })).toHaveAttribute(
      "href",
      "/conversations/conv-1",
    );
    expect(screen.getByRole("link", { name: /PR review/ })).toHaveAttribute(
      "href",
      "/automations/auto-2",
    );
    expect(
      screen.getByText("Daily at 09:00", { exact: false }),
    ).toBeInTheDocument();
  });

  it("previews 10 rows, expands with View more, then offers View All at 20", async () => {
    const automations = Array.from({ length: 20 }, (_, index) =>
      makeAutomation({
        id: `auto-${index + 1}`,
        name: `Automation ${index + 1}`,
      }),
    );
    vi.mocked(AutomationService.getAutomations).mockResolvedValue({
      automations,
      total: 20,
    });
    vi.mocked(AutomationService.getAutomationRuns).mockResolvedValue({
      runs: [makeRun()],
      total: 1,
    });
    const user = userEvent.setup();

    renderHomeAutomations(<RunningAutomationsList />);

    expect(
      await screen.findByTestId("running-automations-list"),
    ).toBeInTheDocument();
    expect(screen.getByText("Automation 1")).toBeInTheDocument();
    expect(screen.getByText("Automation 10")).toBeInTheDocument();
    expect(screen.queryByText("Automation 11")).not.toBeInTheDocument();

    await user.click(screen.getByTestId("home-automations-view-more"));

    expect(screen.getByText("Automation 11")).toBeInTheDocument();
    expect(screen.getByText("Automation 20")).toBeInTheDocument();
    expect(
      screen.queryByTestId("home-automations-view-more"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("home-automations-view-all")).toHaveAttribute(
      "href",
      "/automations",
    );
  });

  it("pins from the row menu into a rich dashboard card with conversation title and failure detail", async () => {
    vi.mocked(AutomationService.getAutomations).mockResolvedValue({
      automations: [
        makeAutomation({ id: "auto-1", name: "Daily digest" }),
        makeAutomation({ id: "auto-2", name: "PR review" }),
      ],
      total: 2,
    });
    vi.mocked(AutomationService.getAutomationRuns).mockImplementation(
      async (id: string) => {
        if (id === "auto-2") {
          return {
            runs: [
              makeRun({
                id: "run-2",
                status: AutomationRunStatus.FAILED,
                conversation_id: null,
                error_detail: "sandbox timeout",
              }),
            ],
            total: 1,
          };
        }
        return { runs: [makeRun()], total: 1 };
      },
    );
    vi.mocked(
      AgentServerConversationService.batchGetAppConversations,
    ).mockResolvedValue([
      makeConversation("conv-1", "Reviewed the release PR"),
    ]);
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

    await user.click(screen.getByTestId("running-automation-menu-auto-1"));
    expect(
      screen.getByTestId("running-automation-run-auto-1"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("running-automation-view-auto-1"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("running-automation-edit-auto-1"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("running-automation-turn-off-auto-1"),
    ).toBeInTheDocument();
    await user.click(screen.getByTestId("running-automation-pin-auto-1"));

    const dashboard = await screen.findByTestId("pinned-automations-dashboard");
    expect(
      within(dashboard).getByTestId("pinned-automation-card-auto-1"),
    ).toBeInTheDocument();
    expect(
      await within(dashboard).findByRole("link", {
        name: "Reviewed the release PR",
      }),
    ).toHaveAttribute("href", "/conversations/conv-1");

    await user.click(screen.getByTestId("running-automation-menu-auto-2"));
    await user.click(screen.getByTestId("running-automation-pin-auto-2"));
    expect(
      await within(dashboard).findByText("sandbox timeout"),
    ).toBeInTheDocument();
    expect(
      within(dashboard).getByText("AUTOMATIONS$DETAIL$NO_CONVERSATION"),
    ).toBeInTheDocument();

    expect(getStoredPinnedIds()).toContain("auto-1");

    await user.click(screen.getByTestId("pinned-automation-menu-auto-1"));
    expect(
      screen.getByTestId("pinned-automation-run-auto-1"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("pinned-automation-view-auto-1"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("pinned-automation-edit-auto-1"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("pinned-automation-turn-off-auto-1"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("pinned-automation-run-auto-1"));
    await waitFor(() => {
      expect(AutomationService.dispatchAutomation).toHaveBeenCalledWith(
        "auto-1",
      );
    });

    await user.click(screen.getByTestId("pinned-automation-menu-auto-1"));
    await user.click(screen.getByTestId("unpin-automation-auto-1"));
    expect(
      screen.queryByTestId("pinned-automation-card-auto-1"),
    ).not.toBeInTheDocument();
  });

  it("shows an error toast when turning an automation off fails", async () => {
    vi.mocked(AutomationService.toggleAutomation).mockRejectedValue(
      new Error("backend unavailable"),
    );
    const user = userEvent.setup();

    renderHomeAutomations(<RunningAutomationsList />);

    await screen.findByTestId("running-automations-list");
    await user.click(screen.getByTestId("running-automation-menu-auto-1"));
    await user.click(screen.getByTestId("running-automation-turn-off-auto-1"));
    await user.click(screen.getByTestId("turn-off-automation-confirm"));

    await waitFor(() => {
      expect(AutomationService.toggleAutomation).toHaveBeenCalledWith(
        "auto-1",
        false,
      );
    });
    await waitFor(() => {
      expect(displayErrorToast).toHaveBeenCalled();
    });
  });
});
