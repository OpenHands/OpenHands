import { http, HttpResponse, delay } from "msw";
import type {
  Automation,
  AutomationsResponse,
  AutomationRun,
  AutomationRunsResponse,
} from "#/types/automation";
import { AutomationRunStatus } from "#/types/automation";
import { MOCK_AUTOMATIONS_RESPONSE } from "./automations.mock";
import { MOCK_AUTOMATION_RUNS } from "./automation-runs.mock";

// Mutable copy for CRUD operations within the mock session
const automations = new Map<string, Automation>(
  MOCK_AUTOMATIONS_RESPONSE.automations.map((a) => [a.id, { ...a }]),
);

export const resetAutomationMockData = () => {
  automations.clear();
  MOCK_AUTOMATIONS_RESPONSE.automations.forEach((a) => {
    automations.set(a.id, { ...a });
  });
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

    const run: AutomationRun = {
      id: crypto.randomUUID(),
      status: AutomationRunStatus.PENDING,
      conversation_id: null,
      bash_command_id: null,
      error_detail: null,
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
