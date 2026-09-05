import { http, HttpResponse } from "msw";
import { FEATURE_DEV_API_PATH } from "#/api/feature-developer-service/feature-developer-constants";
import type {
  FeatureDevRun,
  FeatureDevTicket,
} from "#/api/feature-developer-service/feature-developer-types";

let runs: FeatureDevRun[] = [];
let nextId = 1;

function id(prefix: string): string {
  nextId += 1;
  return `${prefix}-${nextId}`;
}

function now(): string {
  return new Date().toISOString();
}

export function resetFeatureDevMockData() {
  runs = [];
  nextId = 1;
}

function getRun(runId: string): FeatureDevRun | undefined {
  return runs.find((run) => run.id === runId);
}

function ticket(
  runId: string,
  title: string,
  status: FeatureDevTicket["status"],
  extras: Partial<FeatureDevTicket> = {},
): FeatureDevTicket {
  const createdAt = now();
  return {
    id: id("ticket"),
    run_id: runId,
    card_id: id("card"),
    title,
    status,
    branch_name:
      extras.branch_name ?? `feat/${title.toLowerCase().replace(/\s+/g, "-")}`,
    session_id: extras.session_id ?? id("session"),
    estimate_usd: extras.estimate_usd ?? 0.2,
    actual_usd: extras.actual_usd ?? 0.25,
    error: extras.error ?? null,
    started_at: extras.started_at ?? createdAt,
    finished_at: extras.finished_at ?? createdAt,
    position: extras.position ?? 0,
  };
}

export const FEATURE_DEV_HANDLERS = [
  http.get(`*${FEATURE_DEV_API_PATH}`, () => HttpResponse.json(runs)),
  http.post(`*${FEATURE_DEV_API_PATH}`, async ({ request }) => {
    const body = (await request.json()) as {
      project_id?: string;
      spec_text?: string;
      max_concurrent_agents?: number;
      continue_on_failure?: boolean;
    };
    if (!body.project_id?.trim() || !body.spec_text?.trim()) {
      return HttpResponse.json(
        { error: "project_id and spec_text are required" },
        { status: 400 },
      );
    }
    const createdAt = now();
    const runId = id("run");
    const failed = /fail/i.test(body.spec_text);
    const partial = /partial/i.test(body.spec_text);
    let ticketStatus: FeatureDevTicket["status"] = "passed";
    let runStatus: FeatureDevRun["status"] = "passed";
    let error: string | null = null;
    let actualUsd = 0.25;
    if (failed) {
      ticketStatus = "failed";
      runStatus = "failed";
      error = "commit loop failed";
      actualUsd = 0.1;
    } else if (partial) {
      ticketStatus = "failed";
      runStatus = "partial";
      error = "commit loop failed";
      actualUsd = 0.1;
    }
    const tickets = [
      ticket(runId, "Add login form", "passed", { position: 0 }),
      ticket(runId, "Add session cookie", ticketStatus, {
        position: 1,
        error,
        actual_usd: actualUsd,
      }),
    ];
    const run: FeatureDevRun = {
      id: runId,
      project_id: body.project_id.trim(),
      spec_text: body.spec_text.trim(),
      board_id: id("board"),
      status: runStatus,
      current_ticket_index: tickets.length,
      total_estimate_usd: 0.4,
      total_actual_usd: tickets.reduce(
        (sum, item) => sum + Number(item.actual_usd ?? 0),
        0,
      ),
      max_concurrent_agents: body.max_concurrent_agents ?? 1,
      continue_on_failure: Boolean(body.continue_on_failure),
      created_at: createdAt,
      updated_at: createdAt,
      tickets,
    };
    runs.unshift(run);
    return HttpResponse.json(run, { status: 201 });
  }),
  http.get(`*${FEATURE_DEV_API_PATH}/:runId/report`, ({ params }) => {
    const run = getRun(String(params.runId));
    if (!run) {
      return HttpResponse.json({ error: "not found" }, { status: 404 });
    }
    const rows = run.tickets
      .map(
        (item) =>
          `| ${item.title} | ${item.status} | $${Number(item.actual_usd ?? 0).toFixed(4)} | ${item.branch_name ?? ""} |`,
      )
      .join("\n");
    return HttpResponse.json({
      markdown: `| Title | Status | Cost | Branch |\n| --- | --- | --- | --- |\n${rows}\n`,
    });
  }),
  http.post(`*${FEATURE_DEV_API_PATH}/:runId/abort`, ({ params }) => {
    const run = getRun(String(params.runId));
    if (!run) {
      return HttpResponse.json({ error: "not found" }, { status: 404 });
    }
    run.status = "aborted";
    run.updated_at = now();
    return HttpResponse.json(run);
  }),
  http.get(`*${FEATURE_DEV_API_PATH}/:runId`, ({ params }) => {
    const run = getRun(String(params.runId));
    if (!run) {
      return HttpResponse.json({ error: "not found" }, { status: 404 });
    }
    return HttpResponse.json(run);
  }),
];
