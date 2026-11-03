import { beforeEach, describe, expect, it } from "vitest";

import {
  getAutomationFormSession,
  initializeAutomationFormSession,
  patchAutomationFormSession,
} from "#/api/automation-form-session";
import { AUTOMATION_FORM_UPDATE_ACTION_KIND } from "#/constants/automation-form";
import { handleAutomationFormUpdateAction } from "#/services/automation-form";
import { isAutomationFormUpdateActionEvent } from "#/types/agent-server/type-guards";

describe("handleAutomationFormUpdateAction", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });

  it("recognizes automation form update action events", () => {
    expect(
      isAutomationFormUpdateActionEvent({
        id: "event-1",
        timestamp: "2026-01-01T00:00:00.000Z",
        source: "agent",
        tool_name: "automation_form_update",
        tool_call_id: "tool-call-1",
        action: {
          kind: AUTOMATION_FORM_UPDATE_ACTION_KIND,
          fields: { name: "Daily triage" },
        },
      } as never),
    ).toBe(true);
  });

  it("applies agent patches to an existing automation setup draft", () => {
    initializeAutomationFormSession("conv-1", {
      prompt: "Initial automation",
      kind: "prompt",
    });

    const result = handleAutomationFormUpdateAction(
      {
        kind: AUTOMATION_FORM_UPDATE_ACTION_KIND,
        fields: {
          name: "Daily triage",
          prompt: "Summarize incoming issues every morning",
          triggerKind: "cron",
          frequency: "daily",
        },
      },
      "conv-1",
      "event-1",
      "2026-01-01T00:00:00.000Z",
    );

    expect(result).toEqual({
      applied: ["name", "prompt", "triggerKind", "frequency"],
      skipped: [],
      duplicate: false,
    });
    expect(getAutomationFormSession("conv-1")?.form).toMatchObject({
      name: "Daily triage",
      prompt: "Summarize incoming issues every morning",
      frequency: "daily",
    });
    expect(
      getAutomationFormSession("conv-1")?.fieldMetadata?.name,
    ).toMatchObject({
      updatedBy: "agent",
      userDirty: false,
    });
  });

  it("does not replay duplicate agent events", () => {
    initializeAutomationFormSession("conv-1", {
      prompt: "Initial automation",
      kind: "prompt",
    });

    handleAutomationFormUpdateAction(
      {
        kind: AUTOMATION_FORM_UPDATE_ACTION_KIND,
        fields: { name: "First value" },
      },
      "conv-1",
      "event-1",
    );
    const result = handleAutomationFormUpdateAction(
      {
        kind: AUTOMATION_FORM_UPDATE_ACTION_KIND,
        fields: { name: "Second value" },
      },
      "conv-1",
      "event-1",
    );

    expect(result.duplicate).toBe(true);
    expect(getAutomationFormSession("conv-1")?.form?.name).toBe("First value");
  });

  it.each([
    {
      name: "protects user-edited fields by default",
      overwriteUserEdits: undefined,
      expectedResult: { applied: [], skipped: ["name"] },
      expectedName: "Manual title",
    },
    {
      name: "overwrites user-edited fields when explicit",
      overwriteUserEdits: true,
      expectedResult: { applied: ["name"], skipped: [] },
      expectedName: "Agent title",
    },
  ])("$name", ({ overwriteUserEdits, expectedResult, expectedName }) => {
    initializeAutomationFormSession("conv-1", {
      prompt: "Initial automation",
      kind: "prompt",
    });
    patchAutomationFormSession(
      "conv-1",
      { name: "Manual title" },
      { source: "user" },
    );

    const result = handleAutomationFormUpdateAction(
      {
        kind: AUTOMATION_FORM_UPDATE_ACTION_KIND,
        fields: { name: "Agent title" },
        overwrite_user_edits: overwriteUserEdits,
      },
      "conv-1",
      "event-2",
    );

    expect(result).toMatchObject(expectedResult);
    expect(getAutomationFormSession("conv-1")?.form?.name).toBe(expectedName);
  });
});
