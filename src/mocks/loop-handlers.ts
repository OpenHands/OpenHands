import { http, HttpResponse } from "msw";
import {
  LOOPS_API_PATH,
  LOOPS_TRIGGERS_API_PATH,
} from "#/api/loop-service/loop-constants";
import type {
  LoopDefinition,
  LoopRun,
  LoopTrigger,
  TriggerEvent,
} from "#/api/loop-service/loop-types";

let definitions: LoopDefinition[] = [];
let triggers: LoopTrigger[] = [];
let events: TriggerEvent[] = [];
let runs: LoopRun[] = [];
let nextId = 1;

function id(prefix: string): string {
  nextId += 1;
  return `${prefix}-${nextId}`;
}

function now(): string {
  return new Date().toISOString();
}

function definition(idValue: string, name: string): LoopDefinition {
  return {
    id: idValue,
    name,
    project_id: "proj-1",
    stages: [{ name: "run", cmd: null, iterative: false }],
    max_iterations: 10,
    max_cost_usd: 5,
    on_failure: "stop",
    config: {},
    created_at: now(),
    updated_at: now(),
  };
}

export function resetLoopMockData() {
  definitions = [
    definition("def-commit", "commit-loop"),
    definition("def-visual", "visual-regression"),
    definition("def-perf", "perf-loop"),
    definition("def-manual", "manual-loop"),
    definition("def-doc", "doc-loop"),
  ];
  triggers = [];
  events = [];
  runs = [];
  nextId = 1;
}

export function seedLoopTrigger(
  partial: Partial<LoopTrigger> = {},
): LoopTrigger {
  const created: LoopTrigger = {
    id: partial.id ?? id("trigger"),
    project_id: partial.project_id ?? "proj-1",
    loop_definition_id: partial.loop_definition_id ?? "def-commit",
    trigger_type: partial.trigger_type ?? "manual",
    schedule_type: partial.schedule_type ?? null,
    cron_expr: partial.cron_expr ?? null,
    interval_seconds: partial.interval_seconds ?? null,
    payload: partial.payload ?? {},
    enabled: partial.enabled ?? true,
    last_fired_at: partial.last_fired_at ?? null,
    created_at: partial.created_at ?? now(),
    updated_at: partial.updated_at ?? now(),
  };
  triggers.unshift(created);
  return created;
}

export function seedTriggerEvent(
  partial: Partial<TriggerEvent> = {},
): TriggerEvent {
  const created: TriggerEvent = {
    id: partial.id ?? id("event"),
    trigger_id: partial.trigger_id ?? "trigger-1",
    loop_run_id: partial.loop_run_id ?? null,
    fired_at: partial.fired_at ?? now(),
    status: partial.status ?? "fired",
    reason: partial.reason ?? "manual",
  };
  events.unshift(created);
  return created;
}

resetLoopMockData();

export const LOOP_HANDLERS = [
  http.get(`*${LOOPS_TRIGGERS_API_PATH}/events`, () =>
    HttpResponse.json(events),
  ),
  http.get(`*${LOOPS_TRIGGERS_API_PATH}/:triggerId/events`, ({ params }) => {
    const triggerId = String(params.triggerId);
    return HttpResponse.json(
      events.filter((event) => event.trigger_id === triggerId),
    );
  }),
  http.post(`*${LOOPS_TRIGGERS_API_PATH}/:triggerId/fire`, ({ params }) => {
    const trigger = triggers.find(
      (item) => item.id === String(params.triggerId),
    );
    if (!trigger) {
      return HttpResponse.json({ error: "not found" }, { status: 404 });
    }
    const run: LoopRun = {
      id: id("run"),
      definition_id: trigger.loop_definition_id,
      project_id: trigger.project_id,
      session_id: null,
      worktree_dir: null,
      status: "passed",
      current_stage: null,
      iteration: 1,
      total_cost_usd: 0.01,
      created_at: now(),
      updated_at: now(),
      stages: [],
    };
    runs.unshift(run);
    const event: TriggerEvent = {
      id: id("event"),
      trigger_id: trigger.id,
      loop_run_id: run.id,
      fired_at: now(),
      status: "fired",
      reason: "manual",
    };
    events.unshift(event);
    trigger.last_fired_at = event.fired_at;
    return HttpResponse.json({ event, run }, { status: 201 });
  }),
  http.get(`*${LOOPS_TRIGGERS_API_PATH}`, () => HttpResponse.json(triggers)),
  http.post(`*${LOOPS_TRIGGERS_API_PATH}`, async ({ request }) => {
    const body = (await request.json()) as Partial<LoopTrigger>;
    if (
      body.trigger_type === "scheduled" &&
      !body.cron_expr &&
      !body.interval_seconds
    ) {
      return HttpResponse.json(
        { error: "scheduled triggers require a schedule" },
        { status: 400 },
      );
    }
    const created: LoopTrigger = {
      id: id("trigger"),
      project_id: String(body.project_id || "proj-1"),
      loop_definition_id: String(body.loop_definition_id || "def-commit"),
      trigger_type: body.trigger_type || "manual",
      schedule_type: body.schedule_type ?? null,
      cron_expr: body.cron_expr ?? null,
      interval_seconds: body.interval_seconds ?? null,
      payload: body.payload ?? {},
      enabled: body.enabled ?? true,
      last_fired_at: null,
      created_at: now(),
      updated_at: now(),
    };
    triggers.unshift(created);
    return HttpResponse.json(created, { status: 201 });
  }),
  http.patch(
    `*${LOOPS_TRIGGERS_API_PATH}/:triggerId`,
    async ({ params, request }) => {
      const trigger = triggers.find(
        (item) => item.id === String(params.triggerId),
      );
      if (!trigger) {
        return HttpResponse.json({ error: "not found" }, { status: 404 });
      }
      const body = (await request.json()) as Partial<LoopTrigger>;
      Object.assign(trigger, body, { updated_at: now() });
      return HttpResponse.json(trigger);
    },
  ),
  http.get(`*${LOOPS_API_PATH}/runs/:runId`, ({ params }) => {
    const run = runs.find((item) => item.id === String(params.runId));
    if (!run) {
      return HttpResponse.json({ error: "not found" }, { status: 404 });
    }
    return HttpResponse.json(run);
  }),
  http.get(`*${LOOPS_API_PATH}/:definitionId/runs`, ({ params }) => {
    const definitionId = String(params.definitionId);
    return HttpResponse.json(
      runs.filter((run) => run.definition_id === definitionId),
    );
  }),
  http.get(`*${LOOPS_API_PATH}`, () => HttpResponse.json(definitions)),
];
