import { beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import AutomationService from "#/api/automation-service/automation-service.api";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { FeaturedAutomationsSection } from "#/components/features/home/featured-automations/featured-automations-section";
import { NavigationProvider } from "#/context/navigation-context";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
} from "#/types/automation";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
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

function renderSection() {
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
        <FeaturedAutomationsSection />
      </NavigationProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  window.sessionStorage.clear();
  vi.mocked(AutomationService.checkHealth).mockResolvedValue({ status: "ok" });
  vi.mocked(AutomationService.getAutomations).mockResolvedValue({
    automations: [makeAutomation()],
    total: 1,
  });
  vi.mocked(AutomationService.getAutomationRuns).mockResolvedValue({
    runs: [makeRun()],
    total: 1,
  });
  vi.mocked(
    AgentServerConversationService.batchGetAppConversations,
  ).mockResolvedValue([]);
});

describe("FeaturedAutomationsSection", () => {
  it("exposes a chip with accessible last-run health per enabled automation and omits disabled ones", async () => {
    vi.mocked(AutomationService.getAutomations).mockResolvedValue({
      automations: [
        makeAutomation({ id: "auto-1", name: "Daily digest" }),
        makeAutomation({ id: "auto-2", name: "PR review" }),
        makeAutomation({ id: "auto-3", name: "Nightly sweep" }),
        makeAutomation({ id: "auto-4", name: "Repo monitor" }),
        makeAutomation({
          id: "auto-5",
          name: "Disabled sweep",
          enabled: false,
        }),
      ],
      total: 5,
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
        if (id === "auto-3") {
          return {
            runs: [
              makeRun({ id: "run-3", status: AutomationRunStatus.CANCELLED }),
            ],
            total: 1,
          };
        }
        if (id === "auto-4") {
          return { runs: [], total: 0 };
        }
        return { runs: [makeRun()], total: 1 };
      },
    );

    renderSection();

    expect(
      await screen.findByRole("button", {
        name: /Daily digest\s*FEATURED_AUTOMATIONS\$LAST_RUN_SUCCEEDED/,
      }),
    ).toBeInTheDocument();
    expect(
      await screen.findByRole("button", {
        name: /PR review\s*FEATURED_AUTOMATIONS\$LAST_RUN_FAILED/,
      }),
    ).toBeInTheDocument();
    expect(
      await screen.findByRole("button", {
        name: /Nightly sweep\s*FEATURED_AUTOMATIONS\$STATUS_UNKNOWN/,
      }),
    ).toBeInTheDocument();
    expect(
      await screen.findByRole("button", {
        name: /Repo monitor\s*AUTOMATIONS\$DETAIL\$NO_RUNS/,
      }),
    ).toBeInTheDocument();
    expect(screen.queryByText("Disabled sweep")).not.toBeInTheDocument();
  });

  it("renders nothing when the automation service is unavailable", async () => {
    vi.mocked(AutomationService.checkHealth).mockResolvedValue({
      status: "error",
      message: "unreachable",
    });

    renderSection();

    await waitFor(() =>
      expect(AutomationService.checkHealth).toHaveBeenCalledTimes(1),
    );
    expect(
      screen.queryByTestId("featured-automations-section"),
    ).not.toBeInTheDocument();
    expect(AutomationService.getAutomations).not.toHaveBeenCalled();
  });

  it("renders nothing when there are no enabled automations", async () => {
    vi.mocked(AutomationService.getAutomations).mockResolvedValue({
      automations: [makeAutomation({ enabled: false })],
      total: 1,
    });

    renderSection();

    await waitFor(() =>
      expect(AutomationService.getAutomations).toHaveBeenCalledTimes(1),
    );
    expect(
      screen.queryByTestId("featured-automations-section"),
    ).not.toBeInTheDocument();
  });

  it("featuring an automation shows an expanded card linking to the run's conversation", async () => {
    vi.mocked(
      AgentServerConversationService.batchGetAppConversations,
    ).mockResolvedValue([
      makeConversation("conv-1", "Reviewed the release PR"),
    ]);
    const user = userEvent.setup();

    renderSection();
    const chip = await screen.findByRole("button", { name: /Daily digest/ });
    await user.click(chip);

    expect(chip).toHaveAttribute("aria-pressed", "true");
    const conversationLink = await screen.findByRole("link", {
      name: "Reviewed the release PR",
    });
    expect(conversationLink).toHaveAttribute("href", "/conversations/conv-1");
  });

  it("clicking a featured chip again removes its card", async () => {
    const user = userEvent.setup();

    renderSection();
    const chip = await screen.findByRole("button", { name: /Daily digest/ });
    await user.click(chip);
    await screen.findByRole("article");

    await user.click(chip);

    expect(chip).toHaveAttribute("aria-pressed", "false");
    expect(screen.queryByRole("article")).not.toBeInTheDocument();
  });

  it("shows the error detail and missing-conversation note for a failed run", async () => {
    vi.mocked(AutomationService.getAutomationRuns).mockResolvedValue({
      runs: [
        makeRun({
          status: AutomationRunStatus.FAILED,
          conversation_id: null,
          bash_command_id: null,
          error_detail: "Process exited with code 1",
        }),
      ],
      total: 1,
    });
    const user = userEvent.setup();

    renderSection();
    await user.click(
      await screen.findByRole("button", { name: /Daily digest/ }),
    );

    const card = await screen.findByRole("article");
    expect(
      within(card).getByText("Process exited with code 1"),
    ).toBeInTheDocument();
    expect(
      within(card).getByText(/AUTOMATIONS\$DETAIL\$NO_CONVERSATION/),
    ).toBeInTheDocument();
    expect(
      within(card).queryByRole("link", {
        name: "FEATURED_AUTOMATIONS$VIEW_CONVERSATION",
      }),
    ).not.toBeInTheDocument();
  });

  it("keeps the featured selection for the session across remounts", async () => {
    const user = userEvent.setup();

    const { unmount } = renderSection();
    await user.click(
      await screen.findByRole("button", { name: /Daily digest/ }),
    );
    await screen.findByRole("article");
    unmount();

    renderSection();

    const chip = await screen.findByRole("button", { name: /Daily digest/ });
    await waitFor(() => expect(chip).toHaveAttribute("aria-pressed", "true"));
    expect(await screen.findByRole("article")).toBeInTheDocument();
  });

  it("links the add control to the automations page", async () => {
    renderSection();

    const manageLink = await screen.findByRole("link", {
      name: "FEATURED_AUTOMATIONS$MANAGE",
    });
    expect(manageLink).toHaveAttribute("href", "/automations");
  });
});
