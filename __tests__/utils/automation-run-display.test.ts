import { describe, expect, it } from "vitest";
import {
  getAutomationRunBadgeLabelKey,
  getAutomationRunDisplay,
  getAutomationRunTaskOutcome,
} from "#/utils/automation-run-display";
import { AutomationRunStatus, type AutomationRun } from "#/types/automation";
import { I18nKey } from "#/i18n/declaration";

function makeRun(overrides: Partial<AutomationRun> = {}): AutomationRun {
  return {
    id: "run-1",
    status: AutomationRunStatus.COMPLETED,
    conversation_id: "conv-1",
    bash_command_id: "cmd-1",
    error_detail: null,
    started_at: "2026-01-01T10:00:00Z",
    completed_at: "2026-01-01T10:02:00Z",
    ...overrides,
  };
}

describe("automation run display", () => {
  it.each([
    ["success", "success", I18nKey.AUTOMATIONS$DETAIL$SUCCESSFUL],
    ["blocked", "blocked", I18nKey.AUTOMATIONS$DETAIL$BLOCKED],
    ["failed", "failed", I18nKey.AUTOMATIONS$DETAIL$FAILED],
    ["partial_success", "partial_success", I18nKey.AUTOMATIONS$DETAIL$PARTIAL],
    ["unknown", "unknown", I18nKey.AUTOMATIONS$DETAIL$NEEDS_REVIEW],
  ] as const)(
    "uses completed run task outcome %s as the display badge",
    (outcomeStatus, expectedBadge, expectedLabel) => {
      const run = makeRun({
        status: AutomationRunStatus.COMPLETED,
        error_detail: "HUBSPOT_API_KEY was unavailable.",
        run_metadata: {
          finish_tool_response: {
            status: outcomeStatus,
            outcome_summary:
              "Attempted the contact search but the key was missing.",
          },
        },
      });

      const display = getAutomationRunDisplay(run);

      expect(display.badgeStatus).toBe(expectedBadge);
      expect(display.summary).toBe(
        "Attempted the contact search but the key was missing.",
      );
      expect(display.customTaskMetadataText).toBeNull();
      expect(getAutomationRunBadgeLabelKey(display.badgeStatus)).toBe(
        expectedLabel,
      );
    },
  );

  it("defaults a completed run with no task outcome to successful", () => {
    const display = getAutomationRunDisplay(makeRun());

    expect(display.badgeStatus).toBe("success");
    expect(display.summary).toBeNull();
    expect(display.taskOutcome).toBeNull();
  });

  it("keeps lifecycle status for failed infrastructure runs and shows error as row summary", () => {
    const display = getAutomationRunDisplay(
      makeRun({
        status: AutomationRunStatus.FAILED,
        error_detail: "Sandbox timed out.",
      }),
    );

    expect(display.badgeStatus).toBe(AutomationRunStatus.FAILED);
    expect(display.summary).toBe("Sandbox timed out.");
  });

  it("treats an invalid finish tool status as needs review", () => {
    const run = makeRun({
      run_metadata: {
        finish_tool_response: {
          status: "unexpected_status",
          outcome_summary: "Finished but could not classify the task.",
        },
      },
    });

    expect(getAutomationRunTaskOutcome(run)).toEqual({
      status: "unknown",
      outcomeSummary: "Finished but could not classify the task.",
    });
    expect(getAutomationRunDisplay(run).customTaskMetadataText).toContain(
      '"status": "unexpected_status"',
    );
  });

  it("marks completed runs with arbitrary finish metadata as needs review", () => {
    const display = getAutomationRunDisplay(
      makeRun({
        run_metadata: {
          finish_tool_response: {
            crm_contacts_checked: 12,
            matches: ["Acme", "Globex"],
            next_action: "Ask user to pick a contact.",
          },
        },
      }),
    );

    expect(display.badgeStatus).toBe("unknown");
    expect(display.summary).toBeNull();
    expect(display.taskOutcome).toBeNull();
    expect(display.customTaskMetadataText).toContain(
      '"crm_contacts_checked": 12',
    );
    expect(display.customTaskMetadataText).toContain(
      '"next_action": "Ask user to pick a contact."',
    );
  });

  it("keeps default task summary while exposing additional custom metadata", () => {
    const display = getAutomationRunDisplay(
      makeRun({
        run_metadata: {
          finish_tool_response: {
            status: "partial_success",
            outcome_summary: "Posted 4 of 6 items.",
            posted_count: 4,
            skipped_sources: ["feed-a", "feed-b"],
          },
        },
      }),
    );

    expect(display.badgeStatus).toBe("partial_success");
    expect(display.summary).toBe("Posted 4 of 6 items.");
    expect(display.customTaskMetadataText).toContain('"posted_count": 4');
    expect(display.customTaskMetadataText).toContain('"feed-a"');
  });
});
