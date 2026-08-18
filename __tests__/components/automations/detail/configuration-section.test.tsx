import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ConfigurationSection } from "#/components/features/automations/detail/configuration-section";
import type { Automation } from "#/types/automation";

function renderSection(automation: Automation) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <ConfigurationSection automation={automation} />
    </QueryClientProvider>,
  );
}

const cronAutomation: Automation = {
  id: "auto-1",
  name: "Daily digest",
  prompt: "Summarize PRs",
  trigger: {
    type: "cron",
    schedule: "0 9 * * *",
    schedule_human: "Daily at 09:00",
  },
  enabled: true,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  model: "fast-model",
  repository: "acme/app",
  branch: "main",
  timezone: "UTC",
};

const eventAutomation: Automation = {
  id: "auto-2",
  name: "PR Review Bot",
  prompt: "Review PRs",
  trigger: {
    type: "event",
    source: "github",
    on: "pull_request.opened",
    filter: "repository.full_name == 'acme/frontend-app'",
  },
  enabled: true,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  model: "review-model",
  repository: "acme/frontend-app",
  branch: "main",
};

const eventMultiPatternAutomation: Automation = {
  ...eventAutomation,
  id: "auto-3",
  trigger: {
    type: "event",
    source: "github",
    on: ["push", "release.published"],
    filter: "glob(release.tag_name, 'v*')",
  },
};

describe("ConfigurationSection", () => {
  it("renders cron trigger with schedule", () => {
    renderSection(cronAutomation);

    // t() returns the key in tests
    expect(
      screen.getByText("AUTOMATIONS$DETAIL$TRIGGER_SCHEDULE"),
    ).toBeInTheDocument();
    expect(screen.getByText("Daily at 09:00 (UTC)")).toBeInTheDocument();
    expect(screen.getByText("fast-model")).toBeInTheDocument();
    expect(screen.getByText("acme/app")).toBeInTheDocument();
  });

  it("renders event trigger with source, event type, and filter", () => {
    renderSection(eventAutomation);

    expect(
      screen.getByText("AUTOMATIONS$DETAIL$TRIGGER_EVENT"),
    ).toBeInTheDocument();
    expect(screen.getByText("github")).toBeInTheDocument();
    expect(screen.getByText("pull_request.opened")).toBeInTheDocument();
    expect(
      screen.getByText("repository.full_name == 'acme/frontend-app'"),
    ).toBeInTheDocument();
  });

  it("does not show schedule field for event triggers", () => {
    renderSection(eventAutomation);

    expect(
      screen.queryByText("AUTOMATIONS$DETAIL$SCHEDULE"),
    ).not.toBeInTheDocument();
  });

  it("renders multiple event patterns joined by comma", () => {
    renderSection(eventMultiPatternAutomation);

    expect(screen.getByText("push, release.published")).toBeInTheDocument();
  });

  it("shows expand/collapse for long filter expressions", async () => {
    const longFilter =
      "repository.full_name == 'acme/frontend-app' && contains(pull_request.labels[].name, 'needs-review') && sender.login != 'bot'";
    const automation: Automation = {
      ...eventAutomation,
      trigger: {
        type: "event",
        source: "github",
        on: "pull_request.opened",
        filter: longFilter,
      },
    };

    const user = userEvent.setup();
    renderSection(automation);

    expect(screen.getByText("SETTINGS$SKILLS_SHOW_MORE")).toBeInTheDocument();
    expect(screen.queryByText(longFilter)).not.toBeInTheDocument();

    await user.click(screen.getByText("SETTINGS$SKILLS_SHOW_MORE"));
    expect(screen.getByText(longFilter)).toBeInTheDocument();
    expect(screen.getByText("SETTINGS$SKILLS_SHOW_LESS")).toBeInTheDocument();
  });

  it("does not render the filter field when the event trigger has no filter", () => {
    const automation: Automation = {
      ...eventAutomation,
      trigger: {
        type: "event",
        source: "github",
        on: "pull_request.opened",
      },
    };

    renderSection(automation);

    expect(
      screen.queryByText("AUTOMATIONS$DETAIL$EVENT_FILTER"),
    ).not.toBeInTheDocument();
  });

  it("renders the full repo list from preset_metadata when present", () => {
    const automation: Automation = {
      ...cronAutomation,
      preset_metadata: {
        repos: [
          { url: "https://github.com/acme/app", ref: "main" },
          { url: "https://github.com/acme/design-system" },
        ],
      },
    };

    renderSection(automation);

    expect(screen.getByText("acme/app")).toBeInTheDocument();
    expect(screen.getByText("acme/design-system")).toBeInTheDocument();
  });

  it("falls back to the single repository field when preset_metadata has no repos", () => {
    renderSection(cronAutomation);

    expect(screen.getByText("acme/app")).toBeInTheDocument();
  });

  it("does not render a phantom notification field", () => {
    renderSection(cronAutomation);

    expect(
      screen.queryByText("AUTOMATIONS$DETAIL$NOTIFICATION"),
    ).not.toBeInTheDocument();
  });

  it("keeps timeout and keep_alive collapsed by default, revealed after expanding Advanced", async () => {
    const automation: Automation = {
      ...cronAutomation,
      timeout: 900,
      keep_alive: true,
    };
    const user = userEvent.setup();
    renderSection(automation);

    const toggle = screen.getByTestId("configuration-advanced-toggle");
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByTestId("configuration-advanced")).toHaveAttribute(
      "aria-hidden",
      "true",
    );

    await user.click(toggle);

    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByTestId("configuration-advanced")).toHaveAttribute(
      "aria-hidden",
      "false",
    );
    expect(
      screen.getByText("AUTOMATIONS$DETAIL$TIMEOUT_SECONDS"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("AUTOMATIONS$DETAIL$KEEP_ALIVE_ON"),
    ).toBeInTheDocument();
  });

  it("shows the server-default timeout label when timeout is unset", async () => {
    const user = userEvent.setup();
    renderSection(cronAutomation);

    await user.click(screen.getByTestId("configuration-advanced-toggle"));

    expect(
      screen.getByText("AUTOMATIONS$DETAIL$TIMEOUT_DEFAULT"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("AUTOMATIONS$DETAIL$KEEP_ALIVE_OFF"),
    ).toBeInTheDocument();
  });
});
