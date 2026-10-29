import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { ActiveStatusBadge } from "#/components/features/automations/detail/active-status-badge";
import { I18nKey } from "#/i18n/declaration";
import userEvent from "@testing-library/user-event";
import { ConfigurationSection } from "#/components/features/automations/detail/configuration-section";
import { type Automation } from "#/types/automation";
import { DetailHeader } from "#/components/features/automations/detail/detail-header";
import { DisabledReasonBanner } from "#/components/features/automations/detail/disabled-reason-banner";
import { NotFoundState } from "#/components/features/automations/detail/not-found-state";

describe("ActiveStatusBadge", () => {
  it.each([
    [
      { enabled: true },
      I18nKey.AUTOMATIONS$DETAIL$ACTIVE,
      "active-status-badge-active",
    ],
    [
      { enabled: false },
      I18nKey.AUTOMATIONS$DETAIL$INACTIVE,
      "active-status-badge-inactive",
    ],
    [
      { enabled: false, state: "DRAFT" },
      I18nKey.AUTOMATIONS$DETAIL$DRAFT,
      "active-status-badge-draft",
    ],
  ])(
    "renders the matching label and testid for automation %o",
    (automation, labelKey, testId) => {
      render(<ActiveStatusBadge automation={automation} />);

      expect(screen.getByText(labelKey)).toBeInTheDocument();
      expect(screen.getByTestId(testId)).toBeInTheDocument();
    },
  );
});

describe("configuration-section", () => {
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

    it("renders the Automation Runs As field with the resolved identity", () => {
      // Arrange / Act — the route resolves the creator and passes it down.
      render(
        <ConfigurationSection
          automation={cronAutomation}
          runsAs="jdoe@acme.com"
        />,
      );

      // Assert
      expect(
        screen.getByText("AUTOMATIONS$DETAIL$RUNS_AS"),
      ).toBeInTheDocument();
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
});

describe("detail-header", () => {
  const disabledAutomation: Automation = {
    id: "auto-1",
    name: "Daily digest",
    prompt: "Summarize",
    trigger: {
      type: "cron",
      schedule: "0 9 * * *",
      schedule_human: "Daily at 09:00",
    },
    enabled: false,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };

  function renderHeader(
    props: Partial<React.ComponentProps<typeof DetailHeader>> = {},
  ) {
    return render(
      <DetailHeader
        automation={disabledAutomation}
        onToggle={vi.fn()}
        onDelete={vi.fn()}
        onExport={vi.fn()}
        onDownloadTarball={vi.fn()}
        {...props}
      />,
    );
  }

  describe("DetailHeader — status badge", () => {
    it("labels draft automations as draft instead of inactive", () => {
      renderHeader({
        automation: { ...disabledAutomation, state: "DRAFT" },
      });

      expect(screen.getByTestId("active-status-badge-draft")).toHaveTextContent(
        I18nKey.AUTOMATIONS$DETAIL$DRAFT,
      );
      expect(
        screen.queryByTestId("active-status-badge-inactive"),
      ).not.toBeInTheDocument();
    });
  });

  describe("DetailHeader — enabled toggle gating", () => {
    it("hides the switch and Turn on when canToggle is false but keeps Delete", async () => {
      // Arrange — a manager who did not create this disabled automation.
      const user = userEvent.setup();
      renderHeader({ canManage: true, canToggle: false });

      // Act
      await user.click(screen.getByLabelText(I18nKey.AUTOMATIONS$ACTIONS_MENU));

      // Assert
      expect(screen.queryByRole("switch")).not.toBeInTheDocument();
      expect(
        screen.queryByRole("button", { name: I18nKey.AUTOMATIONS$TURN_ON }),
      ).not.toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: I18nKey.AUTOMATIONS$DELETE }),
      ).toBeInTheDocument();
    });

    it("follows canManage for the switch when canToggle is omitted", () => {
      // Arrange / Act
      renderHeader({ canManage: true });

      // Assert
      expect(
        screen.getByRole("switch", { name: I18nKey.AUTOMATIONS$TURN_ON }),
      ).toBeInTheDocument();
    });
  });
});

describe("disabled-reason-banner", () => {
  const baseAutomation: Automation = {
    id: "auto-1",
    name: "Daily digest",
    prompt: "Summarize",
    trigger: { type: "cron", schedule: "0 9 * * *" },
    enabled: false,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };

  describe("DisabledReasonBanner", () => {
    it("renders nothing for an enabled automation even if a stale reason exists", () => {
      const { container } = render(
        <DisabledReasonBanner
          automation={{
            ...baseAutomation,
            enabled: true,
            disabled_reason: "Paused automatically: auth — bad key.",
          }}
        />,
      );
      expect(container.firstChild).toBeNull();
    });

    it("renders nothing for an inactive automation with no recorded reason", () => {
      const { container } = render(
        <DisabledReasonBanner
          automation={{ ...baseAutomation, disabled_reason: null }}
        />,
      );
      expect(container.firstChild).toBeNull();
    });

    it("renders nothing for draft automations even if they carry a disable reason", () => {
      const { container } = render(
        <DisabledReasonBanner
          automation={{
            ...baseAutomation,
            state: "DRAFT",
            disabled_reason: "Automation moved to draft by user",
          }}
        />,
      );

      expect(container.firstChild).toBeNull();
    });

    it("surfaces the automatic-pause reason text and timestamp", () => {
      const reason =
        "Paused automatically: auth — Invalid API key. This failed the last 3 runs and needs a configuration fix.";
      render(
        <DisabledReasonBanner
          automation={{
            ...baseAutomation,
            disabled_reason: reason,
            disabled_detail: {
              reason: "consecutive_permanent_failures",
              source: "consecutive_permanent_failures",
            },
            disabled_at: "2026-09-14T10:00:00Z",
          }}
        />,
      );

      expect(
        screen.getByTestId("automation-disabled-reason-banner"),
      ).toBeInTheDocument();
      expect(
        screen.getByTestId("automation-disabled-reason-text"),
      ).toHaveTextContent(reason);
      expect(
        screen.getByTestId("automation-disabled-reason-timestamp"),
      ).toBeInTheDocument();
    });

    it("hides the timestamp when disabled_at is missing or invalid", () => {
      render(
        <DisabledReasonBanner
          automation={{
            ...baseAutomation,
            disabled_reason:
              "Paused automatically: the last 5 runs all failed.",
            disabled_detail: {
              reason: "consecutive_failures",
              source: "consecutive_failures",
            },
            disabled_at: null,
          }}
        />,
      );

      expect(
        screen.queryByTestId("automation-disabled-reason-timestamp"),
      ).toBeNull();
    });

    it("hides the timestamp for an epoch/zero disabled_at", () => {
      render(
        <DisabledReasonBanner
          automation={{
            ...baseAutomation,
            disabled_reason:
              "Paused automatically: the last 5 runs all failed.",
            disabled_detail: {
              reason: "consecutive_failures",
              source: "consecutive_failures",
            },
            disabled_at: "1970-01-01T00:00:00Z",
          }}
        />,
      );

      expect(
        screen.queryByTestId("automation-disabled-reason-timestamp"),
      ).toBeNull();
    });

    it("substitutes the manual-disable label", () => {
      render(
        <DisabledReasonBanner
          automation={{
            ...baseAutomation,
            disabled_reason: "manual",
            disabled_detail: { reason: "manual", source: "user" },
            disabled_at: null,
          }}
        />,
      );

      expect(
        screen.getByTestId("automation-disabled-reason-text"),
      ).toHaveTextContent(I18nKey.AUTOMATIONS$DETAIL$DISABLED_MANUAL);
      expect(
        screen.queryByTestId("automation-disabled-reason-timestamp"),
      ).toBeNull();
    });
  });
});

describe("NotFoundState", () => {
  it("renders not found state with link back to automations list", () => {
    const { container } = render(<NotFoundState />);

    // Check for the main container
    expect(container.querySelector("div")).toBeInTheDocument();
    // Check for the link to go back
    const link = screen.getByRole("link");
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute("href", "/automations");
  });
});
