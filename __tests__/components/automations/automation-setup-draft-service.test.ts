import { describe, expect, it } from "vitest";
import { HttpError } from "@openhands/typescript-client";
import {
  extractDraftDispatchErrors,
  formFromServerDraft,
  getDraftExecutionStatusText,
  setupDraftFromServerDraft,
} from "#/components/features/automations/setup/automation-setup-draft-service";
import type { AutomationSetupFormValues } from "#/api/automation-setup-types";
import type { AutomationDraftApiResponse } from "#/manifests/types";
import { AutomationRunStatus, type AutomationRun } from "#/types/automation";
import { I18nKey } from "#/i18n/declaration";

const detailEnvelope = {
  detail: {
    message: "Draft is not dispatchable",
    errors: [
      {
        field: "prompt",
        code: "required",
        message: "Prompt is required",
      },
    ],
  },
};

describe("extractDraftDispatchErrors", () => {
  it("reads FastAPI detail envelopes from axios draft dispatch errors", () => {
    const error = {
      isAxiosError: true,
      response: { data: detailEnvelope },
    };

    expect(extractDraftDispatchErrors(error)).toBe("Prompt is required");
  });

  it("reads FastAPI detail envelopes from SDK HttpError draft dispatch errors", () => {
    const error = new HttpError(422, "Unprocessable Entity", detailEnvelope);

    expect(extractDraftDispatchErrors(error)).toBe("Prompt is required");
  });

  it("falls back to the detail message when no field errors are present", () => {
    const error = new HttpError(422, "Unprocessable Entity", {
      detail: { message: "Draft is not dispatchable", errors: [] },
    });

    expect(extractDraftDispatchErrors(error)).toBe("Draft is not dispatchable");
  });
});

const baseSavedDraft: AutomationDraftApiResponse = {
  id: "draft-1",
  endpoint: "/v1/preset/prompt",
  name: "Saved draft",
  draft: {},
  validationErrors: null,
  dispatchable: true,
  sourceAutomationId: null,
  materializedAutomationId: null,
  lastTestRunId: null,
  createdAt: "2026-01-01T00:00:00.000Z",
  updatedAt: "2026-01-01T00:00:00.000Z",
};

const baseForm: AutomationSetupFormValues = {
  name: "",
  prompt: "",
  repository: "",
  pluginSource: "",
  pluginRef: "",
  pluginList: "",
  customCode: "",
  entrypoint: "",
  setupScriptPath: "",
  setupScript: "",
  triggerKind: "cron",
  frequency: "daily",
  time: "09:00",
  scheduleDateTime: "",
  timezone: "UTC",
  weekday: "1",
  customSchedule: "",
  eventSource: "github",
  eventKey: "",
  eventFilter: "",
  model: "",
  agentProfileId: "",
  showTimeout: false,
  timeoutSeconds: "",
  kind: "prompt",
};

describe("setupDraftFromServerDraft", () => {
  it("hydrates all repositories and event keys from saved drafts", () => {
    const saved = {
      ...baseSavedDraft,
      draft: {
        prompt: "Triage issues",
        repos: [{ url: "acme/one" }, { url: "acme/two" }],
        trigger: {
          type: "event",
          source: "github",
          on: ["pull_request", "issues"],
        },
      },
    } satisfies AutomationDraftApiResponse;

    const draft = setupDraftFromServerDraft(saved);

    expect(draft.form?.repository).toBe("acme/one, acme/two");
    expect(draft.form?.eventKey).toBe("pull_request,issues");
  });

  it("updates an existing form with all repositories and event keys", () => {
    const saved = {
      ...baseSavedDraft,
      draft: {
        repos: [{ url: "acme/one" }, { url: "acme/two" }],
        trigger: {
          type: "event",
          source: "github",
          on: ["push", "release.published"],
        },
      },
    } satisfies AutomationDraftApiResponse;

    const form = formFromServerDraft(saved, baseForm);

    expect(form.repository).toBe("acme/one, acme/two");
    expect(form.eventKey).toBe("push,release.published");
  });
});

describe("getDraftExecutionStatusText", () => {
  const t = (key: I18nKey) => key;

  const successfulRun: AutomationRun = {
    id: "run-1",
    status: AutomationRunStatus.COMPLETED,
    conversation_id: null,
    bash_command_id: null,
    error_detail: null,
    started_at: "2026-01-01T00:00:00.000Z",
    completed_at: "2026-01-01T00:01:00.000Z",
  };

  const failedRun: AutomationRun = {
    id: "run-2",
    status: AutomationRunStatus.FAILED,
    conversation_id: null,
    bash_command_id: null,
    error_detail: "Sandbox timed out",
    started_at: "2026-01-01T00:00:00.000Z",
    completed_at: "2026-01-01T00:01:00.000Z",
  };

  const runningRun: AutomationRun = {
    id: "run-3",
    status: AutomationRunStatus.RUNNING,
    conversation_id: null,
    bash_command_id: null,
    error_detail: null,
    started_at: "2026-01-01T00:00:00.000Z",
    completed_at: null,
  };

  it("prioritizes draft validation error over a historical successful test run", () => {
    const invalidDraftWithPastSuccess: AutomationDraftApiResponse = {
      ...baseSavedDraft,
      validationErrors: [
        {
          field: "cron",
          code: "value_error",
          message: "Value error, Invalid cron expression: every day",
        },
      ],
    };

    const status = getDraftExecutionStatusText(
      invalidDraftWithPastSuccess,
      [successfulRun],
      t,
    );

    expect(status).toBe("Value error, Invalid cron expression: every day");
  });

  it("prioritizes draft validation error over a historical failed test run", () => {
    const invalidDraftWithPastFailure: AutomationDraftApiResponse = {
      ...baseSavedDraft,
      validationErrors: [
        {
          field: "cron",
          code: "value_error",
          message: "Value error, Invalid cron expression: every day",
        },
      ],
    };

    const status = getDraftExecutionStatusText(
      invalidDraftWithPastFailure,
      [failedRun],
      t,
    );

    expect(status).toBe("Value error, Invalid cron expression: every day");
  });

  it("returns latest test passed when the draft is valid and the latest run succeeded", () => {
    const validDraft: AutomationDraftApiResponse = {
      ...baseSavedDraft,
      validationErrors: null,
    };

    const status = getDraftExecutionStatusText(validDraft, [successfulRun], t);

    expect(status).toBe(I18nKey.AUTOMATION_SETUP$LATEST_TEST_PASSED);
  });

  it("returns latest test failed when the draft is valid and the latest run failed", () => {
    const validDraft: AutomationDraftApiResponse = {
      ...baseSavedDraft,
      validationErrors: null,
    };

    const status = getDraftExecutionStatusText(validDraft, [failedRun], t);

    expect(status).toBe(I18nKey.AUTOMATION_SETUP$LATEST_TEST_FAILED);
  });

  it("returns latest test running when the draft is valid and the latest run is running", () => {
    const validDraft: AutomationDraftApiResponse = {
      ...baseSavedDraft,
      validationErrors: null,
    };

    const status = getDraftExecutionStatusText(validDraft, [runningRun], t);

    expect(status).toBe(I18nKey.AUTOMATION_SETUP$LATEST_TEST_RUNNING);
  });

  it("returns ready to test when the draft is valid and has no historical runs", () => {
    const validDraft: AutomationDraftApiResponse = {
      ...baseSavedDraft,
      validationErrors: null,
    };

    const status = getDraftExecutionStatusText(validDraft, [], t);

    expect(status).toBe(I18nKey.AUTOMATION_SETUP$READY_TO_TEST);
  });
});

