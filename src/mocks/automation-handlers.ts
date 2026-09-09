import { http, HttpResponse, delay } from "msw";
import capabilitiesFixture from "@openhands/extensions/testing/automations/capabilities.json";
import type {
  AutomationDraftApiResponse,
  AutomationDraftListResponse,
  DeploymentCapabilities,
  DraftValidationError,
  SetupRequestBody,
  ValidateDraftResponse,
} from "#/manifests/types";
import type {
  Automation,
  AutomationsResponse,
  AutomationRun,
  AutomationRunsResponse,
} from "#/types/automation";
import { AutomationRunStatus } from "#/types/automation";
import { MOCK_AUTOMATIONS_RESPONSE } from "./automations.mock";
import { MOCK_AUTOMATION_RUNS } from "./automation-runs.mock";

// The "supported" deployment from the published contract fixtures. Discovery
// and preflight answer with it, so the setup flow runs against the same
// reference data the extensions contract is verified against.
const CAPABILITIES: DeploymentCapabilities = {
  ...capabilitiesFixture.responses.supported.body,
  // The drafts feature (OpenHands/automation PR #417) ships after the pinned
  // extensions package, so the fixture does not advertise it yet. Augment it
  // here so the persisted-draft form flow is exercised in mock mode.
  features: [
    ...(capabilitiesFixture.responses.supported.body.features ?? []),
    "automationDrafts",
  ],
};

interface DraftTrigger {
  type?: string;
  schedule?: string;
  on?: string;
  source?: string;
}

// The schedules the mock can read: "* * * * *" and "*/N * * * *". Anything
// else is assumed to satisfy the deployment minimum — this stands in for the
// service's cron parser rather than reimplementing it.
const STEP_SCHEDULE_PATTERN = /^(?:\*|\*\/(\d+)) \* \* \* \*$/;

function cronIntervalSeconds(schedule: string): number | null {
  const match = STEP_SCHEDULE_PATTERN.exec(schedule.trim());
  if (!match) return null;
  return (match[1] ? Number(match[1]) : 1) * 60;
}

// Event deliveries registered for the mock organization. Distinct from
// CAPABILITIES.eventTypes on purpose: a deployment can support an event type
// that no webhook is registered to deliver, and preflight is what catches
// that gap — the fixtures' event-type-not-delivered scenario depends on one
// supported type staying unregistered here.
const REGISTERED_EVENT_TYPES = CAPABILITIES.eventTypes.filter(
  (type) => type !== "pull_request_review_comment.created",
);

// The deployment checks the contract fixtures record: a cron schedule below
// triggers.cron.minIntervalSeconds, and an event type no webhook delivers.
// Errors address the draft by dotted path, exactly as the fixtures do.
function validateDraftTrigger(
  trigger: DraftTrigger | undefined,
): DraftValidationError[] {
  if (!trigger) return [];

  const cron = CAPABILITIES.triggers.cron;
  if (trigger.type === "cron" && trigger.schedule && cron) {
    const interval = cronIntervalSeconds(trigger.schedule);
    if (interval !== null && interval < cron.minIntervalSeconds) {
      return [
        {
          field: "trigger.schedule",
          code: "interval_too_short",
          message: `Minimum interval for this deployment is ${cron.minIntervalSeconds / 60} minutes.`,
        },
      ];
    }
  }

  if (
    trigger.type === "event" &&
    trigger.on &&
    !REGISTERED_EVENT_TYPES.includes(trigger.on)
  ) {
    const source = trigger.source === "github" ? "GitHub" : trigger.source;
    return [
      {
        field: "trigger.on",
        code: "event_type_not_delivered",
        message: `No ${source} webhook delivering ${trigger.on} is registered for this organization.`,
      },
    ];
  }

  return [];
}

// Mutable copy for CRUD operations within the mock session
const automations = new Map<string, Automation>(
  MOCK_AUTOMATIONS_RESPONSE.automations.map((a) => [a.id, { ...a }]),
);

// Server-backed automation drafts (OpenHands/automation PR #417). Held in
// memory for the mock session; the service treats the draft row as the source
// of truth while editing and materializes it on dispatch.
interface MockDraft {
  id: string;
  endpoint: string;
  name: string | null;
  draft_body: Record<string, unknown>;
  validation_errors: DraftValidationError[] | null;
  dispatchable: boolean;
  source_automation_id: string | null;
  materialized_automation_id: string | null;
  last_test_run_id: string | null;
  created_at: string;
  updated_at: string;
}

const drafts = new Map<string, MockDraft>();

function toDraftResponse(draft: MockDraft): AutomationDraftApiResponse {
  return {
    id: draft.id,
    endpoint: draft.endpoint as AutomationDraftApiResponse["endpoint"],
    name: draft.name,
    draft: draft.draft_body as SetupRequestBody,
    validationErrors: draft.validation_errors,
    dispatchable: draft.dispatchable,
    sourceAutomationId: draft.source_automation_id,
    materializedAutomationId: draft.materialized_automation_id,
    lastTestRunId: draft.last_test_run_id,
    createdAt: draft.created_at,
    updatedAt: draft.updated_at,
  };
}

function validateMockDraft(
  endpoint: string,
  body: Record<string, unknown>,
): DraftValidationError[] {
  const errors: DraftValidationError[] = [];
  const name = body.name;
  if (typeof name !== "string" || !name.trim()) {
    errors.push({
      field: "name",
      code: "value_error",
      message: "A name is required.",
    });
  }
  // Preset drafts need a prompt; the raw endpoint needs a tarball_path instead.
  if (endpoint !== "/v1") {
    const prompt = body.prompt;
    if (typeof prompt !== "string" || !prompt.trim()) {
      errors.push({
        field: "prompt",
        code: "value_error",
        message: "A prompt is required.",
      });
    }
  }
  const trigger = body.trigger as DraftTrigger | undefined;
  errors.push(...validateDraftTrigger(trigger));
  return errors;
}

export const resetAutomationMockData = () => {
  automations.clear();
  MOCK_AUTOMATIONS_RESPONSE.automations.forEach((a) => {
    automations.set(a.id, { ...a });
  });
  drafts.clear();
};

export const AUTOMATION_HANDLERS = [
  // GET /api/automation/health — Health check
  http.get("*/api/automation/health", async () => {
    await delay(100);
    return HttpResponse.json({ status: "ok" });
  }),

  // GET /api/automation/v1 — List automations
  http.get("*/api/automation/v1", async ({ request }) => {
    await delay(300);

    const url = new URL(request.url);
    const limit = Number(url.searchParams.get("limit") ?? "50");
    const offset = Number(url.searchParams.get("offset") ?? "0");

    const all = Array.from(automations.values());
    const page = all.slice(offset, offset + limit);

    const response: AutomationsResponse = {
      automations: page,
      total: all.length,
    };

    return HttpResponse.json(response);
  }),

  // GET /api/automation/v1/capabilities — What this deployment supports
  http.get("*/api/automation/v1/capabilities", async () => {
    await delay(200);
    return HttpResponse.json(CAPABILITIES);
  }),

  // POST /api/automation/v1/validate — Preflight a draft without creating it
  http.post("*/api/automation/v1/validate", async ({ request }) => {
    await delay(200);

    const body = (await request.clone().json()) as {
      automationId?: string;
      endpoint?: string;
      draft?: { trigger?: DraftTrigger };
    };

    const errors = validateDraftTrigger(body.draft?.trigger);
    const response: ValidateDraftResponse = {
      valid: errors.length === 0,
      errors,
    };

    return HttpResponse.json(response);
  }),

  // --- Server-backed automation drafts (OpenHands/automation PR #417) ---

  // POST /api/automation/v1/drafts — Create a persisted draft
  http.post("*/api/automation/v1/drafts", async ({ request }) => {
    await delay(200);
    const body = (await request.clone().json()) as {
      endpoint?: string;
      draft?: Record<string, unknown>;
      name?: string;
      source_automation_id?: string;
    };
    const endpoint = body.endpoint ?? "/v1/preset/prompt";
    const draftBody = body.draft ?? {};
    const now = new Date().toISOString();
    const errors = validateMockDraft(endpoint, draftBody);
    const draft: MockDraft = {
      id: crypto.randomUUID(),
      endpoint,
      name: typeof body.name === "string" && body.name ? body.name : null,
      draft_body: draftBody,
      validation_errors: errors.length > 0 ? errors : null,
      dispatchable: errors.length === 0,
      source_automation_id: body.source_automation_id ?? null,
      materialized_automation_id: null,
      last_test_run_id: null,
      created_at: now,
      updated_at: now,
    };
    drafts.set(draft.id, draft);
    return HttpResponse.json(draft, { status: 201 });
  }),

  // GET /api/automation/v1/drafts — List persisted drafts
  http.get("*/api/automation/v1/drafts", async ({ request }) => {
    await delay(200);
    const url = new URL(request.url);
    const limit = Number(url.searchParams.get("limit") ?? "50");
    const offset = Number(url.searchParams.get("offset") ?? "0");
    const all = Array.from(drafts.values()).sort((a, b) =>
      b.updated_at.localeCompare(a.updated_at),
    );
    const page = all.slice(offset, offset + limit);
    const response: AutomationDraftListResponse = {
      drafts: page.map(toDraftResponse),
      total: all.length,
    };
    return HttpResponse.json(response);
  }),

  // GET /api/automation/v1/drafts/{id} — Get a draft
  http.get("*/api/automation/v1/drafts/:id", async ({ params }) => {
    await delay(200);
    const draft = drafts.get(params.id as string);
    if (!draft) {
      return HttpResponse.json(
        { detail: "Automation draft not found" },
        { status: 404 },
      );
    }
    return HttpResponse.json(draft);
  }),

  // PATCH /api/automation/v1/drafts/{id} — Update a draft
  http.patch("*/api/automation/v1/drafts/:id", async ({ params, request }) => {
    await delay(200);
    const draft = drafts.get(params.id as string);
    if (!draft) {
      return HttpResponse.json(
        { detail: "Automation draft not found" },
        { status: 404 },
      );
    }
    const body = (await request.clone().json()) as {
      endpoint?: string;
      draft?: Record<string, unknown>;
      name?: string;
    };
    if (body.endpoint !== undefined) draft.endpoint = body.endpoint;
    if (body.draft !== undefined) draft.draft_body = body.draft;
    if (body.name !== undefined) draft.name = body.name ? body.name : null;
    const errors = validateMockDraft(draft.endpoint, draft.draft_body);
    draft.validation_errors = errors.length > 0 ? errors : null;
    draft.dispatchable = errors.length === 0;
    draft.updated_at = new Date().toISOString();
    return HttpResponse.json(draft);
  }),

  // DELETE /api/automation/v1/drafts/{id} — Delete a draft
  http.delete("*/api/automation/v1/drafts/:id", async ({ params }) => {
    await delay(200);
    const existed = drafts.delete(params.id as string);
    if (!existed) {
      return HttpResponse.json(
        { detail: "Automation draft not found" },
        { status: 404 },
      );
    }
    return new HttpResponse(null, { status: 204 });
  }),

  // POST /api/automation/v1/drafts/{id}/dispatch — Test-dispatch a draft
  http.post("*/api/automation/v1/drafts/:id/dispatch", async ({ params }) => {
    await delay(200);
    const draft = drafts.get(params.id as string);
    if (!draft) {
      return HttpResponse.json(
        { detail: "Automation draft not found" },
        { status: 404 },
      );
    }
    const errors = validateMockDraft(draft.endpoint, draft.draft_body);
    if (errors.length > 0) {
      draft.validation_errors = errors;
      draft.dispatchable = false;
      return HttpResponse.json(
        { message: "Draft is not dispatchable", errors },
        { status: 422 },
      );
    }
    draft.validation_errors = null;
    draft.dispatchable = true;
    draft.updated_at = new Date().toISOString();
    const run: AutomationRun = {
      id: crypto.randomUUID(),
      status: AutomationRunStatus.PENDING,
      conversation_id: crypto.randomUUID(),
      bash_command_id: null,
      error_detail: null,
      started_at: new Date().toISOString(),
      completed_at: null,
    };
    draft.last_test_run_id = run.id;
    return HttpResponse.json(run, { status: 201 });
  }),

  // POST /api/automation/v1/preset/:kind — Create a prompt/plugin automation
  http.post("*/api/automation/v1/preset/:kind", async ({ params, request }) => {
    await delay(200);

    const body = (await request.clone().json()) as {
      name: string;
      prompt: string;
      model?: string;
      trigger: Automation["trigger"];
      repos?: { url: string; ref?: string }[];
      plugins?: { source: string }[];
      timeout?: number;
    };
    const now = new Date().toISOString();
    const automation: Automation = {
      id: crypto.randomUUID(),
      name: body.name,
      prompt: body.prompt,
      model: body.model ?? null,
      trigger: body.trigger,
      enabled: true,
      created_at: now,
      updated_at: now,
      last_triggered_at: null,
      ...(body.repos?.[0] && {
        repository: body.repos[0].url,
        branch: body.repos[0].ref,
      }),
      ...(body.plugins && {
        plugins: body.plugins.map((plugin) => plugin.source),
      }),
      ...(typeof body.timeout === "number" && { timeout: body.timeout }),
      ...(typeof body.trigger.timezone === "string" && {
        timezone: body.trigger.timezone,
      }),
    };

    if (params.kind !== "prompt" && params.kind !== "plugin") {
      return HttpResponse.json(
        { detail: "Unknown preset kind" },
        { status: 404 },
      );
    }

    automations.set(automation.id, automation);
    return HttpResponse.json(automation, { status: 201 });
  }),

  // GET /api/automation/v1/:id/runs — List automation runs
  http.get("*/api/automation/v1/:id/runs", async ({ params, request }) => {
    await delay(200);

    const id = params.id as string;
    if (!automations.has(id)) {
      return HttpResponse.json(
        { detail: "Automation not found" },
        { status: 404 },
      );
    }

    const url = new URL(request.url);
    const limit = Number(url.searchParams.get("limit") ?? "50");
    const offset = Number(url.searchParams.get("offset") ?? "0");
    const allRuns = MOCK_AUTOMATION_RUNS[id] ?? [];
    const page = allRuns.slice(offset, offset + limit);
    const statusCounts: AutomationRunsResponse["status_counts"] = {};
    for (const run of allRuns) {
      statusCounts[run.status] = (statusCounts[run.status] ?? 0) + 1;
    }

    const response: AutomationRunsResponse = {
      runs: page,
      total: allRuns.length,
      status_counts: statusCounts,
    };

    return HttpResponse.json(response);
  }),

  // GET /api/automation/v1/:id — Get automation detail
  http.get("*/api/automation/v1/:id", async ({ params }) => {
    await delay(200);

    const automation = automations.get(params.id as string);
    if (!automation) {
      return HttpResponse.json(
        { detail: "Automation not found" },
        { status: 404 },
      );
    }

    return HttpResponse.json(automation);
  }),

  // PATCH /api/automation/v1/:id — Update automation (toggle enabled)
  http.patch("*/api/automation/v1/:id", async ({ params, request }) => {
    await delay(200);

    const id = params.id as string;
    // Clone the request before reading the body to avoid "Body has already been read" errors
    // when MSW internally consumes the body during handler resolution.
    const body = (await request.clone().json()) as Partial<Automation>;
    const automation = automations.get(id);
    if (!automation) {
      return HttpResponse.json(
        { detail: "Automation not found" },
        { status: 404 },
      );
    }

    const updated: Automation = {
      ...automation,
      ...body,
      updated_at: new Date().toISOString(),
    };
    automations.set(id, updated);

    return HttpResponse.json(updated);
  }),

  // POST /api/automation/v1/:id/dispatch — Manually trigger a run
  http.post("*/api/automation/v1/:id/dispatch", async ({ params }) => {
    await delay(200);

    const id = params.id as string;
    const automation = automations.get(id);
    if (!automation) {
      return HttpResponse.json(
        { detail: "Automation not found" },
        { status: 404 },
      );
    }

    const run: AutomationRun = {
      id: crypto.randomUUID(),
      status: AutomationRunStatus.PENDING,
      conversation_id: null,
      bash_command_id: null,
      error_detail: null,
      // A freshly dispatched run has not reported a phase yet — the service
      // sends the fields as null rather than omitting them.
      phase_code: null,
      phase_label: null,
      phase_updated_at: null,
      started_at: new Date().toISOString(),
      completed_at: null,
    };

    return HttpResponse.json(run, { status: 201 });
  }),

  // DELETE /api/automation/v1/:id — Delete automation
  http.delete("*/api/automation/v1/:id", async ({ params }) => {
    await delay(200);

    const id = params.id as string;
    if (!automations.has(id)) {
      return HttpResponse.json(
        { detail: "Automation not found" },
        { status: 404 },
      );
    }

    automations.delete(id);
    return new HttpResponse(null, { status: 204 });
  }),
];
