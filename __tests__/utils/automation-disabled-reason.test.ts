import { describe, expect, it, vi } from "vitest";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";
import {
  getDisablementReasonDisplay,
  hasDisablementReason,
  isManualDisable,
} from "#/utils/automation-disabled-reason";

const t = (key: I18nKey) => key as string;
const identity = vi.fn((key: I18nKey) => key as string);

function automation(
  overrides: Partial<Pick<Automation, "enabled" | "disabled_reason" | "disabled_detail">>,
) {
  return { enabled: true, ...overrides } as Automation;
}

describe("hasDisablementReason", () => {
  it("is false for an enabled automation even with a stale reason", () => {
    expect(
      hasDisablementReason(
        automation({
          enabled: true,
          disabled_reason: "Paused automatically: auth — bad key.",
        }),
      ),
    ).toBe(false);
  });

  it("is false for an inactive automation with no recorded reason", () => {
    expect(
      hasDisablementReason(automation({ enabled: false, disabled_reason: null })),
    ).toBe(false);
  });

  it("is true when inactive and a reason is recorded", () => {
    expect(
      hasDisablementReason(
        automation({
          enabled: false,
          disabled_reason: "Paused automatically: auth — bad key.",
        }),
      ),
    ).toBe(true);
  });
});

describe("isManualDisable", () => {
  it("detects the manual reason value", () => {
    expect(
      isManualDisable(
        automation({
          disabled_reason: "manual",
          disabled_detail: { reason: "manual", source: "user" },
        }),
      ),
    ).toBe(true);
  });

  it("detects the manual_delete reason value", () => {
    expect(
      isManualDisable(automation({ disabled_reason: "manual_delete" })),
    ).toBe(true);
  });

  it("detects a user source in detail even with a non-manual reason string", () => {
    expect(
      isManualDisable(
        automation({
          disabled_reason: "some-other-reason",
          disabled_detail: { source: "user" },
        }),
      ),
    ).toBe(true);
  });

  it("is false for automatic disablement sources", () => {
    expect(
      isManualDisable(
        automation({
          disabled_reason:
            "Paused automatically: the last 5 runs all failed and nothing has succeeded in 2 days.",
          disabled_detail: { reason: "consecutive_failures", source: "consecutive_failures" },
        }),
      ),
    ).toBe(false);
  });
});

describe("getDisablementReasonDisplay", () => {
  it("returns null when there is no recorded disablement reason", () => {
    expect(
      getDisablementReasonDisplay(automation({ enabled: false, disabled_reason: null }), identity),
    ).toBeNull();
    expect(
      getDisablementReasonDisplay(automation({ enabled: true, disabled_reason: "manual" }), identity),
    ).toBeNull();
  });

  it("substitutes a localized label for manual disables instead of the raw 'manual' string", () => {
    const display = getDisablementReasonDisplay(
      automation({
        enabled: false,
        disabled_reason: "manual",
        disabled_detail: { reason: "manual", source: "user" },
      }),
      identity,
    );
    expect(display).not.toBeNull();
    expect(display?.text).toBe(I18nKey.AUTOMATIONS$DETAIL$DISABLED_MANUAL);
  });

  it("surfaces the backend's human-readable reason for automatic disables", () => {
    const reason =
      "Paused automatically: auth — Invalid API key. This failed the last 3 runs and needs a configuration fix.";
    const display = getDisablementReasonDisplay(
      automation({
        enabled: false,
        disabled_reason: reason,
        disabled_detail: { reason: "consecutive_permanent_failures", source: "consecutive_permanent_failures" },
      }),
      identity,
    );
    expect(display?.text).toBe(reason);
  });

  it("treats an automatic disable with an unknown source as automatic", () => {
    const reason = "Paused automatically: the last 5 runs all failed.";
    const display = getDisablementReasonDisplay(
      automation({ enabled: false, disabled_reason: reason, disabled_detail: null }),
      t,
    );
    expect(display?.text).toBe(reason);
  });
});
