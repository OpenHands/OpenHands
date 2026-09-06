import { http, HttpResponse } from "msw";
import {
  ROUTING_AUDIT_PATH,
  ROUTING_CONFIG_PATH,
  ROUTING_INGEST_PATH,
  ROUTING_LOCAL_RUNTIMES_PATH,
  ROUTING_REGISTRY_PATH,
  ROUTING_RESOLVE_PATH,
  ROUTING_ROUTER_MODEL_PATH,
  ROUTING_SOURCES_PATH,
  ROUTING_TAXONOMY_PATH,
  ROUTING_TARGET_AUTO,
} from "#/api/routing-service/routing-constants";
import type {
  RoutingAuditItem,
  RoutingConfig,
  RoutingGuardrails,
  RoutingLabel,
  RoutingLocalRuntimes,
  RoutingRegistryModel,
  RoutingRegistrySnapshot,
  RoutingResolveRequest,
  RoutingResolveResult,
  RoutingRouterModelResponse,
  RoutingSourceStatus,
  RoutingTaxonomy,
} from "#/api/routing-service/routing-types";

const GUARDRAILS: RoutingGuardrails = {
  forbid_training_retention: false,
  forbid_watermarking: false,
  max_cost_usd_per_task: null,
  max_latency_s: null,
};

const WORK_TYPES: RoutingLabel[] = [
  {
    id: "coding",
    name: "Coding",
    description: "Implement features and fix bugs.",
  },
  {
    id: "review",
    name: "Review",
    description: "Review diffs and pull requests.",
  },
  {
    id: "ux",
    name: "UX + visual design",
    description: "Layout, CSS, and UI polish.",
  },
  {
    id: "copy",
    name: "Copy / marketing",
    description: "Marketing text and UI copy.",
  },
  { id: "docs", name: "Docs", description: "README and API docs." },
  { id: "test", name: "Test", description: "Unit and integration tests." },
  {
    id: "refactor",
    name: "Refactor",
    description: "Restructure without behavior change.",
  },
  { id: "research", name: "Research", description: "Investigate options." },
  { id: "ops", name: "Ops", description: "CI, deploy, and infrastructure." },
];

const SENSITIVITIES: RoutingLabel[] = [
  { id: "public", name: "Public", description: "Already public." },
  { id: "default", name: "Default", description: "Ordinary internal work." },
  {
    id: "sensitive",
    name: "Sensitive",
    description: "Secrets or customer data.",
  },
  {
    id: "sensitive-ip",
    name: "Sensitive IP / proprietary code",
    description: "Proprietary source that must not be retained or watermarked.",
  },
];

function classifierPrompt(
  workTypes: RoutingLabel[],
  sensitivities: RoutingLabel[],
): string {
  const workLines = workTypes
    .map((item) => `- ${item.id}: ${item.name} — ${item.description}`)
    .join("\n");
  const sensLines = sensitivities
    .map((item) => `- ${item.id}: ${item.name} — ${item.description}`)
    .join("\n");
  return `Classify the task into exactly one work_type id, one sensitivity id, and a complexity of low, medium, or high.\nWork types:\n${workLines}\n\nSensitivities:\n${sensLines}\n`;
}

function defaultConfig(): RoutingConfig {
  return {
    goal: "quality",
    guardrails: { ...GUARDRAILS },
    mode: "warn",
    metadata_max_age_days: 90,
    blend_alpha: 0,
    cost_mode: "floor",
    quality_floor: 0.45,
    struggle_threshold: 2,
    router_model: {
      preset: "cheapest",
      disabled: false,
      provider_key: "openhands",
      model: "openhands/glm-5.2",
      goal: "cost",
      guardrails: { ...GUARDRAILS },
    },
    routes: [
      {
        id: "route-sensitive-ip",
        work_type: null,
        sensitivity: "sensitive-ip",
        goal: "privacy",
        target: ROUTING_TARGET_AUTO,
        guardrails: {
          ...GUARDRAILS,
          forbid_training_retention: true,
          forbid_watermarking: true,
        },
      },
      {
        id: "route-ux",
        work_type: "ux",
        sensitivity: null,
        goal: "quality",
        target: ROUTING_TARGET_AUTO,
        guardrails: { ...GUARDRAILS },
      },
      {
        id: "route-copy",
        work_type: "copy",
        sensitivity: null,
        goal: "quality",
        target: ROUTING_TARGET_AUTO,
        guardrails: { ...GUARDRAILS },
      },
      {
        id: "route-default",
        work_type: null,
        sensitivity: null,
        goal: "quality",
        target: ROUTING_TARGET_AUTO,
        guardrails: { ...GUARDRAILS },
      },
    ],
    work_types: WORK_TYPES.map((item) => ({ ...item })),
    sensitivities: SENSITIVITIES.map((item) => ({ ...item })),
    imported_project_path: null,
  };
}

function model(
  partial: Partial<RoutingRegistryModel> &
    Pick<RoutingRegistryModel, "id" | "provider_key">,
): RoutingRegistryModel {
  return {
    benchmarks: {
      coding: { score: 0.7, provenance: null },
      reasoning: { score: 0.65, provenance: null },
    },
    cost_per_1k: 0.002,
    latency_s_p90: 20,
    local: false,
    runtime: null,
    source_url: "https://example.com",
    verified: false,
    notes: null,
    retention: "none",
    watermark: "none",
    reachable: true,
    ...partial,
  };
}

let config = defaultConfig();
let ingestShouldFail = false;
let audit: RoutingAuditItem[] = [];

export function setRoutingIngestFailure(value: boolean) {
  ingestShouldFail = value;
}

export function resetRoutingMockData() {
  config = defaultConfig();
  ingestShouldFail = false;
  audit = [];
}

resetRoutingMockData();

function taxonomy(): RoutingTaxonomy {
  return {
    work_types: config.work_types,
    sensitivities: config.sensitivities,
    classifier_prompt: classifierPrompt(
      config.work_types,
      config.sensitivities,
    ),
  };
}

function registry(): RoutingRegistrySnapshot {
  const models: RoutingRegistryModel[] = [
    model({
      id: "openhands/glm-5.2",
      provider_key: "openhands",
      cost_per_1k: 0.0012,
      benchmarks: {
        coding: { score: 0.78, provenance: null },
        reasoning: { score: 0.75, provenance: null },
      },
    }),
    model({
      id: "anthropic/claude-sonnet-4-5",
      provider_key: "anthropic",
      cost_per_1k: 0.006,
      verified: true,
      benchmarks: {
        coding: {
          score: 0.8,
          provenance: {
            source: "swebench",
            fetched_at: "2026-09-06T00:00:00Z",
            version: "swebench",
          },
        },
        reasoning: { score: 0.84, provenance: null },
      },
    }),
    model({
      id: "google/gemini-2.5-pro",
      provider_key: "gemini",
      retention: "opt-out",
      watermark: "always",
      reachable: false,
    }),
    model({
      id: "ollama/qwen3-coder:16b",
      provider_key: "ollama",
      local: true,
      runtime: "ollama",
      cost_per_1k: 0,
      reachable: true,
    }),
    model({
      id: "ollama/llama3.2:3b",
      provider_key: "ollama",
      local: true,
      runtime: "ollama",
      cost_per_1k: 0,
      reachable: false,
    }),
  ];
  return {
    version: "v1",
    last_updated: "2026-09-06T00:00:00Z",
    privacy_last_updated: "2026-09-06T00:00:00Z",
    stale: false,
    privacy_stale: false,
    models,
    sources: {},
    curated_fields:
      "retention and watermark are manually researched and versioned; no public benchmark source publishes them",
  };
}

function sources(): RoutingSourceStatus[] {
  return [
    {
      id: "swebench",
      last_success: "2026-09-06T00:00:00Z",
      last_error: null,
      stale: false,
    },
    {
      id: "lmsys-arena",
      last_success: null,
      last_error: ingestShouldFail ? "network down" : null,
      stale: ingestShouldFail,
    },
    {
      id: "artificial-analysis",
      last_success: null,
      last_error: null,
      stale: false,
    },
    {
      id: "livebench",
      last_success: null,
      last_error: null,
      stale: false,
    },
  ];
}

function localRuntimes(): RoutingLocalRuntimes {
  return {
    runtimes: {
      ollama: {
        alive: true,
        models: ["qwen3-coder:16b"],
        error: null,
      },
    },
    best_by_category: {
      coding: {
        id: "ollama/qwen3-coder:16b",
        provider_key: "ollama",
        score: 0.55,
      },
    },
  };
}

function routerModel(): RoutingRouterModelResponse {
  const snap = registry();
  const pool = snap.models.map((item) => ({
    id: item.id,
    provider_key: item.provider_key,
    cost_per_1k: item.cost_per_1k,
    latency_s_p90: item.latency_s_p90,
    retention: item.retention,
    watermark: item.watermark,
    classification_accuracy_proxy: 0.7,
    offline_capable: item.local,
    verified: item.verified,
  }));
  const localPool = pool.filter((item) => item.offline_capable);
  const preset = config.router_model.preset;
  const chosen =
    preset === "local"
      ? localPool[0]
      : (pool.find((item) => item.provider_key === "openhands") ?? pool[0]);
  return {
    config: config.router_model,
    resolved: {
      preset,
      provider_key: chosen.provider_key,
      model: chosen.id,
      goal: config.router_model.goal,
      guardrails: config.router_model.guardrails,
      tradeoffs: {
        cost_per_1k: chosen.cost_per_1k,
        latency_s_p90: chosen.latency_s_p90,
        retention: chosen.retention,
        watermark: chosen.watermark,
        classification_accuracy_proxy: chosen.classification_accuracy_proxy,
        offline_capable: chosen.offline_capable,
        verified: chosen.verified,
      },
      pool: preset === "local" ? localPool : pool,
    },
    runtimes: localRuntimes(),
  };
}

function resolve(body: RoutingResolveRequest): RoutingResolveResult {
  const decision = {
    provider_key: "openhands",
    model: "openhands/glm-5.2",
    score: 0.78,
    score_source: "benchmark:coding",
    rule_id: "route-default",
    registry_version: "v1",
    target: ROUTING_TARGET_AUTO as typeof ROUTING_TARGET_AUTO,
    usable: true,
    goal: "quality" as const,
    locked: false,
  };
  const result: RoutingResolveResult = {
    decision,
    trace: {
      task_text: body.task_text,
      classification: {
        work_type: body.work_type ?? "coding",
        sensitivity: body.sensitivity ?? "default",
        complexity: "medium",
        confidence: 0.8,
        reason: "mock classification",
        classifier: "fallback",
        classifier_version: "v1-taxonomy",
      },
      classifier_version: "v1-taxonomy",
      route_id: "route-default",
      filters: [
        {
          id: "google/gemini-2.5-pro",
          reason: "provider gemini is not connected",
        },
      ],
      ranked: [
        {
          id: "openhands/glm-5.2",
          provider_key: "openhands",
          score: 0.78,
          score_source: "benchmark:coding",
        },
        {
          id: "anthropic/claude-sonnet-4-5",
          provider_key: "anthropic",
          score: 0.8,
          score_source: "benchmark:coding",
        },
      ],
      chosen: decision,
      reason: "Auto-picked openhands/openhands/glm-5.2 for the task.",
    },
    audit_id: "audit-1",
  };
  audit.unshift({
    id: "audit-1",
    created_at: new Date().toISOString(),
    kind: "resolve",
    card_id: body.card_id ?? null,
    run_id: body.run_id ?? null,
    payload: { decision: result.decision, trace: result.trace },
  });
  return result;
}

export const ROUTING_HANDLERS = [
  http.get(`*${ROUTING_CONFIG_PATH}`, () => HttpResponse.json(config)),
  http.put(`*${ROUTING_CONFIG_PATH}`, async ({ request }) => {
    const body = (await request.json()) as Partial<RoutingConfig>;
    config = { ...config, ...body };
    if (body.guardrails) {
      config.guardrails = { ...config.guardrails, ...body.guardrails };
    }
    if (body.router_model) {
      config.router_model = { ...config.router_model, ...body.router_model };
    }
    return HttpResponse.json(config);
  }),
  http.get(`*${ROUTING_TAXONOMY_PATH}`, () => HttpResponse.json(taxonomy())),
  http.put(`*${ROUTING_TAXONOMY_PATH}`, async ({ request }) => {
    const body = (await request.json()) as Partial<RoutingTaxonomy> & {
      reset?: boolean;
    };
    if (body.reset) {
      const fresh = defaultConfig();
      config.work_types = fresh.work_types;
      config.sensitivities = fresh.sensitivities;
    } else {
      if (body.work_types) config.work_types = body.work_types;
      if (body.sensitivities) config.sensitivities = body.sensitivities;
    }
    return HttpResponse.json(taxonomy());
  }),
  http.get(`*${ROUTING_REGISTRY_PATH}`, () => HttpResponse.json(registry())),
  http.get(`*${ROUTING_SOURCES_PATH}`, () =>
    HttpResponse.json({ sources: sources() }),
  ),
  http.post(`*${ROUTING_INGEST_PATH}`, () => {
    if (ingestShouldFail) {
      return HttpResponse.json(
        {
          sources: { swebench: { ok: false, error: "network down" } },
          unmapped: [],
        },
        { status: 200 },
      );
    }
    return HttpResponse.json({
      sources: { swebench: { ok: true, rows: 2, unmapped: 1 } },
      unmapped: [{ name: "mystery-lab-9", source: "swebench" }],
    });
  }),
  http.get(`*${ROUTING_ROUTER_MODEL_PATH}`, () =>
    HttpResponse.json(routerModel()),
  ),
  http.put(`*${ROUTING_ROUTER_MODEL_PATH}`, async ({ request }) => {
    const body = (await request.json()) as Partial<
      RoutingConfig["router_model"]
    >;
    config.router_model = { ...config.router_model, ...body };
    return HttpResponse.json(routerModel());
  }),
  http.get(`*${ROUTING_LOCAL_RUNTIMES_PATH}`, () =>
    HttpResponse.json(localRuntimes()),
  ),
  http.post(`*${ROUTING_RESOLVE_PATH}`, async ({ request }) => {
    const body = (await request.json()) as RoutingResolveRequest;
    return HttpResponse.json(resolve(body));
  }),
  http.get(`*${ROUTING_AUDIT_PATH}`, () =>
    HttpResponse.json({
      items: audit,
      total: audit.length,
      limit: 50,
      offset: 0,
    }),
  ),
];
