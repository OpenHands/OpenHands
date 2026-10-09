import { describe, expect, it } from "vitest";
import { HttpError } from "@openhands/typescript-client";
import {
  extractDraftDispatchErrors,
  formFromServerDraft,
  setupDraftFromServerDraft,
} from "#/components/features/automations/setup/automation-setup-draft-service";
import type { AutomationSetupFormValues } from "#/api/automation-setup-types";
import type { AutomationDraftApiResponse } from "#/manifests/types";

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
