import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DisabledReasonBanner } from "#/components/features/automations/detail/disabled-reason-banner";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";

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
      <DisabledReasonBanner automation={{ ...baseAutomation, disabled_reason: null }} />,
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
          disabled_detail: { reason: "consecutive_permanent_failures", source: "consecutive_permanent_failures" },
          disabled_at: "2026-09-14T10:00:00Z",
        }}
      />,
    );

    expect(screen.getByTestId("automation-disabled-reason-banner")).toBeInTheDocument();
    expect(screen.getByTestId("automation-disabled-reason-text")).toHaveTextContent(
      reason,
    );
    expect(screen.getByTestId("automation-disabled-reason-timestamp")).toBeInTheDocument();
  });

  it("hides the timestamp when disabled_at is missing or invalid", () => {
    render(
      <DisabledReasonBanner
        automation={{
          ...baseAutomation,
          disabled_reason: "Paused automatically: the last 5 runs all failed.",
          disabled_detail: { reason: "consecutive_failures", source: "consecutive_failures" },
          disabled_at: null,
        }}
      />,
    );

    expect(screen.queryByTestId("automation-disabled-reason-timestamp")).toBeNull();
  });

  it("hides the timestamp for an epoch/zero disabled_at", () => {
    render(
      <DisabledReasonBanner
        automation={{
          ...baseAutomation,
          disabled_reason: "Paused automatically: the last 5 runs all failed.",
          disabled_detail: { reason: "consecutive_failures", source: "consecutive_failures" },
          disabled_at: "1970-01-01T00:00:00Z",
        }}
      />,
    );

    expect(screen.queryByTestId("automation-disabled-reason-timestamp")).toBeNull();
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

    expect(screen.getByTestId("automation-disabled-reason-text")).toHaveTextContent(
      I18nKey.AUTOMATIONS$DETAIL$DISABLED_MANUAL,
    );
    expect(screen.queryByTestId("automation-disabled-reason-timestamp")).toBeNull();
  });
});
