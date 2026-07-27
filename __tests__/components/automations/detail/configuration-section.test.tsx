import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ConfigurationSection } from "#/components/features/automations/detail/configuration-section";
import type { Automation } from "#/types/automation";

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
    render(<ConfigurationSection automation={cronAutomation} />);

    // t() returns the key in tests
    expect(
      screen.getByText("AUTOMATIONS$DETAIL$TRIGGER_SCHEDULE"),
    ).toBeInTheDocument();
    expect(screen.getByText("Daily at 09:00 (UTC)")).toBeInTheDocument();
    expect(screen.getByText("fast-model")).toBeInTheDocument();
    expect(screen.getByText("acme/app")).toBeInTheDocument();
  });

  it("renders event trigger with source, event type, and filter", () => {
    render(<ConfigurationSection automation={eventAutomation} />);

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
    render(<ConfigurationSection automation={eventAutomation} />);

    expect(
      screen.queryByText("AUTOMATIONS$DETAIL$SCHEDULE"),
    ).not.toBeInTheDocument();
  });

  it("renders multiple event patterns joined by comma", () => {
    render(<ConfigurationSection automation={eventMultiPatternAutomation} />);

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
    render(<ConfigurationSection automation={automation} />);

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

    render(<ConfigurationSection automation={automation} />);

    expect(
      screen.queryByText("AUTOMATIONS$DETAIL$EVENT_FILTER"),
    ).not.toBeInTheDocument();
  });
});
