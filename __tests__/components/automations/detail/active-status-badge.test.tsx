import userEvent from "@testing-library/user-event";
import { type Automation } from "#/types/automation";
import { DetailHeader } from "#/components/features/automations/detail/detail-header";
import { DisabledReasonBanner } from "#/components/features/automations/detail/disabled-reason-banner";
import { NotFoundState } from "#/components/features/automations/detail/not-found-state";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { automationListPath } from "#/manifests/automation-interface";
import { ActiveStatusBadge } from "#/components/features/automations/detail/active-status-badge";
import { I18nKey } from "#/i18n/declaration";
import {
  LIGHT_THEMES,
  PANEL_SURFACES,
  colorUtility,
  contrast,
  over,
  surfaceColor,
  utilityColor,
} from "../../../themes/theme-color-resolver";

describe("ActiveStatusBadge", () => {
  it.each(LIGHT_THEMES)(
    "keeps the rendered draft badge readable on %s panels",
    (theme) => {
      render(
        <ActiveStatusBadge automation={{ enabled: false, state: "DRAFT" }} />,
      );
      const classes = screen.getByTestId("active-status-badge-draft").className;
      const fill = utilityColor(theme, colorUtility(classes, "bg"));
      const ink = utilityColor(theme, colorUtility(classes, "text"));
      for (const surface of PANEL_SURFACES) {
        const background = over(fill, surfaceColor(theme, surface));
        expect
          .soft(contrast(over(ink, background), background), surface)
          .toBeGreaterThanOrEqual(4.5);
      }
    },
  );
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

  describe("DetailHeader — back arrow", () => {
    it("returns to the automations list", () => {
      renderHeader();

      expect(
        screen.getByRole("link", { name: I18nKey.COMMON$BACK }),
      ).toHaveAttribute(
        "href",
        expect.stringMatching(new RegExp(`${automationListPath()}$`)),
      );
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

  describe("DetailHeader — keep sandbox alive", () => {
    const enabledAutomation: Automation = {
      ...disabledAutomation,
      enabled: true,
    };

    it.each([
      { keepAlive: false, label: I18nKey.AUTOMATIONS$KEEP_SANDBOX_ALIVE },
      {
        keepAlive: true,
        label: I18nKey.AUTOMATIONS$STOP_KEEPING_SANDBOX_ALIVE,
      },
    ])(
      "lists $label between Turn off and Delete when keep_alive is $keepAlive",
      async ({ keepAlive, label }) => {
        // Arrange
        const user = userEvent.setup();
        const onToggleKeepAlive = vi.fn();
        renderHeader({
          automation: { ...enabledAutomation, keep_alive: keepAlive },
          onToggleKeepAlive,
        });
        await user.click(
          screen.getByLabelText(I18nKey.AUTOMATIONS$ACTIONS_MENU),
        );
        const names = screen
          .getAllByRole("button")
          .map((button) => button.textContent);
        expect(
          names.slice(names.indexOf(I18nKey.AUTOMATIONS$TURN_OFF)),
        ).toEqual([
          I18nKey.AUTOMATIONS$TURN_OFF,
          label,
          I18nKey.AUTOMATIONS$DELETE,
        ]);

        // Act
        await user.click(screen.getByRole("button", { name: label }));

        // Assert
        expect(onToggleKeepAlive).toHaveBeenCalledTimes(1);
      },
    );

    it("disables the item while a keep_alive save is pending", async () => {
      // Arrange
      const user = userEvent.setup();
      renderHeader({
        automation: { ...enabledAutomation, keep_alive: false },
        onToggleKeepAlive: vi.fn(),
        isKeepAlivePending: true,
      });

      // Act
      await user.click(screen.getByLabelText(I18nKey.AUTOMATIONS$ACTIONS_MENU));

      // Assert
      expect(
        screen.getByRole("button", {
          name: I18nKey.AUTOMATIONS$KEEP_SANDBOX_ALIVE,
        }),
      ).toBeDisabled();
    });

    it("leaves the item out when onToggleKeepAlive is omitted", async () => {
      // Arrange
      const user = userEvent.setup();
      renderHeader({ automation: { ...enabledAutomation, keep_alive: false } });

      // Act
      await user.click(screen.getByLabelText(I18nKey.AUTOMATIONS$ACTIONS_MENU));

      // Assert
      expect(
        screen.queryByRole("button", {
          name: I18nKey.AUTOMATIONS$KEEP_SANDBOX_ALIVE,
        }),
      ).not.toBeInTheDocument();
    });
  });

  describe("DetailHeader — edit gating", () => {
    it("hides Edit when canEdit is false but keeps Delete", async () => {
      // Arrange — a manager viewing a automation that runs as a person.
      const user = userEvent.setup();
      renderHeader({ canManage: true, canEdit: false, onEdit: vi.fn() });

      // Act
      await user.click(screen.getByLabelText(I18nKey.AUTOMATIONS$ACTIONS_MENU));

      // Assert
      expect(
        screen.queryByRole("button", { name: I18nKey.AUTOMATIONS$EDIT }),
      ).not.toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: I18nKey.AUTOMATIONS$DELETE }),
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
});
