import { http, HttpResponse, delay } from "msw";
import capabilitiesFixture from "@openhands/extensions/testing/automations/capabilities.json";
import type {
  DeploymentCapabilities,
  DraftValidationError,
  ValidateDraftResponse,
} from "#/manifests/types";
import type {
  Automation,
  AutomationsResponse,
  AutomationRun,
  AutomationRunsResponse,
} from "#/types/automation";
import { AutomationRunStatus } from "#/types/automation";
import type {
  CustomWebhook,
  CustomWebhookCreateResponse,
  CustomWebhookListResponse,
} from "#/types/webhook";
import { MOCK_AUTOMATIONS_RESPONSE } from "./automations.mock";
import { MOCK_AUTOMATION_RUNS } from "./automation-runs.mock";

// The "supported" deployment from the published contract fixtures. Discovery
// and preflight answer with it, so the setup flow runs against the same
// reference data the extensions contract is verified against.
const CAPABILITIES: DeploymentCapabilities =
  capabilitiesFixture.responses.supported.body;

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

// Runs are seeded from MOCK_AUTOMATION_RUNS but mutated in place (status
// transitions on cancel), so keep a mutable per-session copy the same way
// `automations` does.
const runsById = new Map<string, AutomationRun>();

function resetRunsMockData() {
  runsById.clear();
  Object.values(MOCK_AUTOMATION_RUNS)
    .flat()
    .forEach((run) => runsById.set(run.id, { ...run }));
}
resetRunsMockData();

export const resetAutomationMockData = () => {
  automations.clear();
  MOCK_AUTOMATIONS_RESPONSE.automations.forEach((a) => {
    automations.set(a.id, { ...a });
  });
  resetRunsMockData();
};

const webhooks = new Map<string, CustomWebhook & { webhook_secret: string }>();

export const resetWebhookMockData = () => {
  webhooks.clear();
};

function generateMockSecret(): string {
  return `whsec_${crypto.randomUUID().replace(/-/g, "")}`;
}

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

  // POST /api/automation/v1/preset/:kind — Create a prompt/plugin automation
  http.post("*/api/automation/v1/preset/:kind", async ({ params, request }) => {
    await delay(200);

    const body = (await request.clone().json()) as {
      name: string;
      prompt: string;
      model?: string;
      trigger: Automation["trigger"];
      repos?: { url: string; ref?: string; provider?: string }[];
      plugins?: { source: string }[];
      variants?: unknown[];
      timeout?: number;
      keep_alive?: boolean;
    };
    const now = new Date().toISOString();
    const presetMetadata =
      body.repos?.length || body.plugins?.length
        ? {
            ...(body.repos?.length && {
              repos: body.repos.map((r) => ({
                url: r.url,
                ...(r.ref && { ref: r.ref }),
                ...(r.provider && {
                  provider: r.provider as "github" | "gitlab" | "bitbucket",
                }),
              })),
            }),
            ...(body.plugins?.length && {
              plugins: body.plugins.map((p) => p.source),
            }),
          }
        : undefined;

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
      ...(presetMetadata && { preset_metadata: presetMetadata }),
      ...(body.plugins && {
        plugins: body.plugins.map((plugin) => plugin.source),
      }),
      ...(typeof body.timeout === "number" && { timeout: body.timeout }),
      ...(typeof body.keep_alive === "boolean" && {
        keep_alive: body.keep_alive,
      }),
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
    const seedIds = (MOCK_AUTOMATION_RUNS[id] ?? []).map((run) => run.id);
    const allRuns = seedIds
      .map((runId) => runsById.get(runId))
      .filter((run): run is AutomationRun => !!run);
    const page = allRuns.slice(offset, offset + limit);

    const response: AutomationRunsResponse = {
      runs: page,
      total: allRuns.length,
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

    const now = new Date().toISOString();
    const run: AutomationRun = {
      id: crypto.randomUUID(),
      automation_id: id,
      status: AutomationRunStatus.PENDING,
      conversation_id: null,
      bash_command_id: null,
      sandbox_id: null,
      error_detail: null,
      created_at: now,
      started_at: now,
      completed_at: null,
      timeout_at: automation.timeout
        ? new Date(Date.now() + automation.timeout * 1000).toISOString()
        : null,
    };
    runsById.set(run.id, run);

    return HttpResponse.json(run, { status: 201 });
  }),

  // POST /api/automation/v1/runs/:runId/cancel — Cancel a pending/running run
  http.post("*/api/automation/v1/runs/:runId/cancel", async ({ params }) => {
    await delay(200);

    const runId = params.runId as string;
    const run = runsById.get(runId);
    if (!run) {
      return HttpResponse.json({ detail: "Run not found" }, { status: 404 });
    }

    if (
      run.status !== AutomationRunStatus.PENDING &&
      run.status !== AutomationRunStatus.RUNNING
    ) {
      return HttpResponse.json(
        {
          detail: `Run is ${run.status}, only PENDING or RUNNING runs can be cancelled`,
        },
        { status: 409 },
      );
    }

    const cancelled: AutomationRun = {
      ...run,
      status: AutomationRunStatus.CANCELLED,
      completed_at: new Date().toISOString(),
      error_detail: "Cancelled by user",
    };
    runsById.set(runId, cancelled);

    return HttpResponse.json(cancelled);
  }),

  // --- Custom webhooks (org-scoped, /v1/webhooks) ---

  // GET /api/automation/v1/webhooks — List custom webhooks
  http.get("*/api/automation/v1/webhooks", async ({ request }) => {
    await delay(200);

    const url = new URL(request.url);
    const limit = Number(url.searchParams.get("limit") ?? "50");
    const offset = Number(url.searchParams.get("offset") ?? "0");
    const all = Array.from(webhooks.values()).map(
      ({ webhook_secret: _secret, ...rest }) => rest,
    );
    const page = all.slice(offset, offset + limit);

    const response: CustomWebhookListResponse = {
      webhooks: page,
      total: all.length,
    };
    return HttpResponse.json(response);
  }),

  // POST /api/automation/v1/webhooks — Create a custom webhook
  http.post("*/api/automation/v1/webhooks", async ({ request }) => {
    await delay(200);

    const body = (await request.clone().json()) as {
      name: string;
      source: string;
      event_key_expr?: string;
      signature_header?: string;
      webhook_secret?: string;
    };

    const existing = Array.from(webhooks.values()).find(
      (w) => w.source === body.source,
    );
    if (existing) {
      return HttpResponse.json(
        { detail: `Webhook source '${body.source}' already exists` },
        { status: 409 },
      );
    }

    const now = new Date().toISOString();
    const generatedSecret = body.webhook_secret ? null : generateMockSecret();
    const id = crypto.randomUUID();
    const webhook: CustomWebhook & { webhook_secret: string } = {
      id,
      org_id: "org-mock",
      name: body.name,
      source: body.source,
      webhook_url: `/api/automation/v1/events/org-mock/${body.source}`,
      event_key_expr: body.event_key_expr ?? "type",
      signature_header: body.signature_header ?? "X-Signature-256",
      enabled: true,
      created_at: now,
      updated_at: now,
      webhook_secret: body.webhook_secret ?? generatedSecret ?? "",
    };
    webhooks.set(id, webhook);

    const { webhook_secret: _stored, ...publicFields } = webhook;
    const response: CustomWebhookCreateResponse = {
      ...publicFields,
      ...(generatedSecret && { webhook_secret: generatedSecret }),
    };
    return HttpResponse.json(response, { status: 201 });
  }),

  // PATCH /api/automation/v1/webhooks/:id — Update a webhook
  http.patch(
    "*/api/automation/v1/webhooks/:id",
    async ({ params, request }) => {
      await delay(200);

      const id = params.id as string;
      const webhook = webhooks.get(id);
      if (!webhook) {
        return HttpResponse.json(
          { detail: "Webhook not found" },
          { status: 404 },
        );
      }

      const body = (await request.clone().json()) as Partial<CustomWebhook>;
      const updated = {
        ...webhook,
        ...body,
        updated_at: new Date().toISOString(),
      };
      webhooks.set(id, updated);

      const { webhook_secret: _secret, ...publicFields } = updated;
      return HttpResponse.json(publicFields);
    },
  ),

  // DELETE /api/automation/v1/webhooks/:id — Delete a webhook
  http.delete("*/api/automation/v1/webhooks/:id", async ({ params }) => {
    await delay(200);

    const id = params.id as string;
    if (!webhooks.has(id)) {
      return HttpResponse.json(
        { detail: "Webhook not found" },
        { status: 404 },
      );
    }
    webhooks.delete(id);
    return new HttpResponse(null, { status: 204 });
  }),

  // POST /api/automation/v1/webhooks/:id/rotate-secret — Rotate secret
  http.post(
    "*/api/automation/v1/webhooks/:id/rotate-secret",
    async ({ params }) => {
      await delay(200);

      const id = params.id as string;
      const webhook = webhooks.get(id);
      if (!webhook) {
        return HttpResponse.json(
          { detail: "Webhook not found" },
          { status: 404 },
        );
      }

      const newSecret = generateMockSecret();
      webhooks.set(id, { ...webhook, webhook_secret: newSecret });
      return HttpResponse.json({ webhook_secret: newSecret });
    },
  ),

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
