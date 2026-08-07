import { describe, expect, it, vi } from "vitest";
import { buildHomeAutomationMenuItems } from "#/components/features/home/featured-automations/build-home-automation-menu-items";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";

function makeAutomation(overrides: Partial<Automation> = {}): Automation {
  return {
    id: "auto-1",
    name: "Daily digest",
    prompt: "Summarize",
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

function itemKeys(
  entries: ReturnType<typeof buildHomeAutomationMenuItems>,
): string[] {
  return entries
    .filter((entry) => entry.kind === "item")
    .map((entry) => entry.key);
}

describe("buildHomeAutomationMenuItems", () => {
  it("orders run / view / edit / turn off / pin with a separator", () => {
    const entries = buildHomeAutomationMenuItems({
      automation: makeAutomation(),
      t: (key) => key,
      canManage: true,
      canEdit: true,
      isRunPending: false,
      isCancelPending: false,
      canCancel: false,
      testIdPrefix: "running-automation",
      pinTestId: "running-automation-pin-auto-1",
      isPinned: false,
      onRunNow: vi.fn(),
      onCancelRun: vi.fn(),
      onView: vi.fn(),
      onEdit: vi.fn(),
      onTurnOff: vi.fn(),
      onTogglePin: vi.fn(),
    });

    expect(itemKeys(entries)).toEqual([
      "run-now",
      "view",
      "edit",
      "turn-off",
      "pin",
    ]);
    expect(entries.some((entry) => entry.kind === "separator")).toBe(true);
    const pinItem = entries.find(
      (entry) => entry.kind === "item" && entry.key === "pin",
    );
    expect(pinItem?.kind === "item" && pinItem.label).toBe(
      I18nKey.FEATURED_AUTOMATIONS$PIN,
    );
  });

  it("adds cancel run for in-flight automations and disables run now", () => {
    const entries = buildHomeAutomationMenuItems({
      automation: makeAutomation(),
      t: (key) => key,
      canManage: true,
      canEdit: false,
      isRunPending: false,
      isCancelPending: false,
      canCancel: true,
      testIdPrefix: "pinned-automation",
      pinTestId: "unpin-automation-auto-1",
      isPinned: true,
      onRunNow: vi.fn(),
      onCancelRun: vi.fn(),
      onView: vi.fn(),
      onTurnOff: vi.fn(),
      onTogglePin: vi.fn(),
    });

    expect(itemKeys(entries)).toContain("cancel-run");
    const runNow = entries.find(
      (entry) => entry.kind === "item" && entry.key === "run-now",
    );
    expect(runNow?.kind === "item" && runNow.disabled).toBe(true);
  });

  it("omits manage actions when the user cannot manage automations", () => {
    const entries = buildHomeAutomationMenuItems({
      automation: makeAutomation(),
      t: (key) => key,
      canManage: false,
      canEdit: true,
      isRunPending: false,
      isCancelPending: false,
      canCancel: false,
      testIdPrefix: "running-automation",
      pinTestId: "running-automation-pin-auto-1",
      isPinned: false,
      onRunNow: vi.fn(),
      onCancelRun: vi.fn(),
      onView: vi.fn(),
      onEdit: vi.fn(),
      onTurnOff: vi.fn(),
      onTogglePin: vi.fn(),
    });

    expect(itemKeys(entries)).toEqual(["view", "pin"]);
  });
});
