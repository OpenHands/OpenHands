import { describe, expect, it } from "vitest";
import type { Automation } from "#/types/automation";
import { setupDraftFromAutomation } from "#/utils/automation-edit-draft";

function automation(overrides: Partial<Automation> = {}): Automation {
  return {
    id: "auto-1",
    name: "Daily digest",
    prompt: "Summarize yesterday's PRs",
    trigger: {
      type: "cron",
      schedule: "0 9 * * *",
      timezone: "America/New_York",
    },
    enabled: true,
    repository: "acme/repo",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

describe("setupDraftFromAutomation", () => {
  it("maps a daily prompt automation onto the setup form and keeps the edit id", () => {
    const draft = setupDraftFromAutomation(automation());

    expect(draft.editingAutomationId).toBe("auto-1");
    expect(draft.kind).toBe("prompt");
    expect(draft.form).toMatchObject({
      name: "Daily digest",
      prompt: "Summarize yesterday's PRs",
      repository: "acme/repo",
      frequency: "daily",
      time: "09:00",
      timezone: "America/New_York",
      triggerKind: "cron",
    });
  });

  it("recognizes hourly cron and event triggers", () => {
    expect(
      setupDraftFromAutomation(
        automation({ trigger: { type: "cron", schedule: "0 * * * *" } }),
      ).form?.frequency,
    ).toBe("hourly");

    expect(
      setupDraftFromAutomation(
        automation({
          prompt: "On each pull request",
          trigger: {
            type: "event",
            source: "github",
            on: "pull_request",
            filter: "action == 'opened'",
          },
        }),
      ).form,
    ).toMatchObject({
      triggerKind: "event",
      eventSource: "github",
      eventKey: "pull_request",
      eventFilter: "action == 'opened'",
    });
  });
});
