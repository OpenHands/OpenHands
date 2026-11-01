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
  repositories: [{ url: "acme/app", ref: "main" }],
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
  repositories: [{ url: "acme/frontend-app", ref: "main" }],
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

  it("renders every repository as owner/repo with its branch badge", () => {
    const automation: Automation = {
      ...eventAutomation,
      repositories: [
        { url: "https://github.com/qa-example/qa-repo", ref: "main" },
        { url: "git@gitlab.com:qa-example/docs.git", ref: "v2" },
        { url: "qa-example/tools" },
        {
          url: "https://bitbucket.example.com/scm/DEM/demo-app.git",
          provider: "bitbucket_data_center",
        },
        {
          url: "https://example.com/bitbucket/scm/~ALICE/notes.git",
          provider: "bitbucket_data_center",
        },
      ],
    };

    render(<ConfigurationSection automation={automation} />);

    expect(
      screen.getByText("AUTOMATIONS$DETAIL$REPOSITORIES"),
    ).toBeInTheDocument();
    const rows = screen.getAllByTestId("automation-repository");
    expect(rows.map((row) => row.textContent)).toEqual([
      "qa-example/qa-repomain",
      "qa-example/docsv2",
      "qa-example/tools",
      "DEM/demo-app",
      "~ALICE/notes",
    ]);
  });

  it("omits the repositories field when the automation has none", () => {
    render(
      <ConfigurationSection
        automation={{ ...eventAutomation, repositories: undefined }}
      />,
    );

    expect(
      screen.queryByText("AUTOMATIONS$DETAIL$REPOSITORIES"),
    ).not.toBeInTheDocument();
  });

  it("renders plugin-spec source, configuration, and timeout", () => {
    const automation: Automation = {
      ...cronAutomation,
      timeout: 300,
      preset_metadata: { preset_type: "plugin_spec" },
      plugin_spec: {
        plugin: {
          source: "github:openhands/pr-reviewer",
          ref: "main",
          repo_path: "plugins/pr-reviewer",
        },
        config: { event: "pull_request.opened", max_results: 12 },
      },
    };

    render(<ConfigurationSection automation={automation} />);

    expect(screen.getByTestId("automation-plugin-spec")).toHaveTextContent(
      "github:openhands/pr-reviewer",
    );
    expect(screen.getByTestId("automation-plugin-spec")).toHaveTextContent(
      "plugins/pr-reviewer",
    );
    expect(screen.getByTestId("automation-plugin-config")).toHaveTextContent(
      "pull_request.opened",
    );
    expect(screen.getByTestId("automation-timeout")).toHaveTextContent("300");
  });

  it("expands long plugin configuration values", async () => {
    const longValue = "qa-" + "x".repeat(120);
    const automation: Automation = {
      ...cronAutomation,
      preset_metadata: { preset_type: "plugin_spec" },
      plugin_spec: {
        plugin: { source: "github:openhands/pr-reviewer" },
        config: { description: longValue },
      },
    };

    const user = userEvent.setup();
    render(<ConfigurationSection automation={automation} />);

    expect(
      screen.getByTestId("automation-plugin-config"),
    ).not.toHaveTextContent(longValue);
    await user.click(screen.getByText("SETTINGS$SKILLS_SHOW_MORE"));
    expect(screen.getByTestId("automation-plugin-config")).toHaveTextContent(
      longValue,
    );
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

  it("renders plugin definition when its configuration is empty", () => {
    const automation: Automation = {
      ...cronAutomation,
      preset_metadata: { preset_type: "plugin_spec" },
      plugin_spec: {
        plugin: { source: "github:openhands/pr-reviewer" },
        config: {},
      },
    };

    render(<ConfigurationSection automation={automation} />);

    expect(screen.getByTestId("automation-plugin-spec")).toHaveTextContent(
      "github:openhands/pr-reviewer",
    );
    expect(
      screen.queryByTestId("automation-plugin-config"),
    ).not.toBeInTheDocument();
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

  it("renders the Automation Runs As field with the resolved identity", () => {
    // Arrange / Act — the route resolves the creator and passes it down.
    render(
      <ConfigurationSection
        automation={cronAutomation}
        runsAs="jdoe@acme.com"
      />,
    );

    // Assert
    expect(screen.getByText("AUTOMATIONS$DETAIL$RUNS_AS")).toBeInTheDocument();
    expect(screen.getByText("jdoe@acme.com")).toBeInTheDocument();
  });

  it("does not render the Automation Runs As field when no identity is provided", () => {
    // Arrange / Act — local backends and in-flight lookups pass nothing.
    render(<ConfigurationSection automation={cronAutomation} />);

    // Assert
    expect(
      screen.queryByText("AUTOMATIONS$DETAIL$RUNS_AS"),
    ).not.toBeInTheDocument();
  });
});
