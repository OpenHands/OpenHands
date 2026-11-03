import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TriggerSection } from "#/components/features/automations/detail/trigger-section";
import type { Automation } from "#/types/automation";

const base = {
  name: "Automation",
  prompt: "Do the work",
  enabled: true,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

const cronAutomation: Automation = {
  ...base,
  id: "auto-1",
  trigger: { type: "cron", schedule: "0 9 * * *" },
  timezone: "UTC",
};

const eventAutomation: Automation = {
  ...base,
  id: "auto-2",
  trigger: {
    type: "event",
    source: "github",
    on: ["push", "release.published"],
    filter: "repository.full_name == 'acme/frontend-app'",
  },
};

describe("TriggerSection", () => {
  it("words a schedule and keeps its timezone and cron underneath", () => {
    render(<TriggerSection automation={cronAutomation} />);

    expect(screen.getByTestId("automation-trigger-summary")).toHaveTextContent(
      "AUTOMATIONS$SCHEDULE$DAILY_AT",
    );
    expect(screen.getByText("UTC")).toBeInTheDocument();
    expect(screen.getByText("0 9 * * *")).toBeInTheDocument();
  });

  it("names an event's source and events, then its filter", () => {
    render(<TriggerSection automation={eventAutomation} />);

    expect(screen.getByTestId("automation-trigger-summary")).toHaveTextContent(
      "GitHub · push, release.published",
    );
    expect(
      screen.getByText("repository.full_name == 'acme/frontend-app'"),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("AUTOMATIONS$DETAIL$CRON_EXPRESSION"),
    ).not.toBeInTheDocument();
  });

  it("omits the filter field when the event trigger has none", () => {
    render(
      <TriggerSection
        automation={{
          ...eventAutomation,
          trigger: { type: "event", source: "github", on: "push" },
        }}
      />,
    );

    expect(
      screen.queryByText("AUTOMATIONS$DETAIL$EVENT_FILTER"),
    ).not.toBeInTheDocument();
  });

  it("shows expand/collapse for long filter expressions", async () => {
    const longFilter =
      "repository.full_name == 'acme/frontend-app' && contains(pull_request.labels[].name, 'needs-review') && sender.login != 'bot'";
    const user = userEvent.setup();
    render(
      <TriggerSection
        automation={{
          ...eventAutomation,
          trigger: { ...eventAutomation.trigger, filter: longFilter },
        }}
      />,
    );

    expect(screen.queryByText(longFilter)).not.toBeInTheDocument();

    await user.click(screen.getByText("SETTINGS$SKILLS_SHOW_MORE"));
    expect(screen.getByText(longFilter)).toBeInTheDocument();
    expect(screen.getByText("SETTINGS$SKILLS_SHOW_LESS")).toBeInTheDocument();
  });
});
