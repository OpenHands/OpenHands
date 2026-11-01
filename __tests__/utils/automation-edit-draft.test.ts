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
    // What the client reads: repositories lifted from the service's preset
    // metadata, which has no top-level `repository`.
    repositories: [{ url: "acme/repo", provider: "github" }],
    preset_metadata: {
      preset_type: "prompt",
      repos: [{ url: "acme/repo", provider: "github" }],
    },
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

  it.each([
    {
      name: "hourly cron",
      automation: automation({
        trigger: { type: "cron", schedule: "0 * * * *" },
      }),
      expected: { triggerKind: "cron", frequency: "hourly" },
    },
    {
      name: "custom cron",
      automation: automation({
        trigger: { type: "cron", schedule: "*/17 * * * *" },
      }),
      expected: {
        triggerKind: "cron",
        frequency: "custom",
        customSchedule: "*/17 * * * *",
      },
    },
    {
      name: "event trigger",
      automation: automation({
        prompt: "On each pull request",
        trigger: {
          type: "event",
          source: "github",
          on: "pull_request",
          filter: "action == 'opened'",
        },
      }),
      expected: {
        triggerKind: "event",
        eventSource: "github",
        eventKey: "pull_request",
        eventFilter: "action == 'opened'",
      },
    },
  ])("maps $name fields", ({ automation: savedAutomation, expected }) => {
    expect(setupDraftFromAutomation(savedAutomation).form).toMatchObject(
      expected,
    );
  });

  it("keeps the saved repositories so an edit can tell what changed", () => {
    const draft = setupDraftFromAutomation(
      automation({
        repositories: [
          { url: "acme/repo", ref: "main", provider: "github" },
          { url: "https://github.com/acme/docs" },
        ],
      }),
    );

    expect(draft.form?.repository).toBe(
      "acme/repo, https://github.com/acme/docs",
    );
    expect(draft.existingRepositories).toEqual([
      { url: "acme/repo", ref: "main", provider: "github" },
      { url: "https://github.com/acme/docs" },
    ]);
    expect(draft.canUpdateRepositories).toBe(true);
  });

  it("falls back to the single repository of an imported file", () => {
    const draft = setupDraftFromAutomation(
      automation({ repositories: undefined, repository: "acme/legacy" }),
    );

    expect(draft.form?.repository).toBe("acme/legacy");
  });

  it.each([
    { name: "a plugin spec", preset_metadata: { preset_type: "plugin_spec" } },
    { name: "a custom bundle", preset_metadata: null },
  ])("does not offer repository changes for $name", ({ preset_metadata }) => {
    expect(
      setupDraftFromAutomation(automation({ preset_metadata }))
        .canUpdateRepositories,
    ).toBe(false);
  });
});
