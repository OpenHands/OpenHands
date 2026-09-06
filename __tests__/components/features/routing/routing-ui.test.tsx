import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import RoutingService from "#/api/routing-service/routing-service.api";
import type {
  RoutingAuditItem,
  RoutingConfig,
  RoutingLocalRuntimes,
  RoutingRegistrySnapshot,
  RoutingRouterModelResponse,
  RoutingSourcesResponse,
  RoutingTrace,
} from "#/api/routing-service/routing-types";
import { RoutesTable } from "#/components/features/routing/routes-table";
import { HowAutoRoutes } from "#/components/features/routing/how-auto-routes";
import { DecisionTraceDrawer } from "#/components/features/routing/decision-trace-drawer";
import { DryRunConsole } from "#/components/features/routing/dry-run-console";
import { GoalGuardrails } from "#/components/features/routing/goal-guardrails";
import { TaxonomyEditor } from "#/components/features/routing/taxonomy-editor";
import { RouterModelChooser } from "#/components/features/routing/router-model-chooser";
import { SwitchHistory } from "#/components/features/routing/switch-history";
import { BenchmarkSources } from "#/components/features/routing/benchmark-sources";
import { ModelRegistryBrowser } from "#/components/features/routing/model-registry";
import { RoutingPage } from "#/components/features/routing/routing-page";
import { I18nKey } from "#/i18n/declaration";
import * as toastHandlers from "#/utils/custom-toast-handlers";

const GUARDRAILS = {
  forbid_training_retention: false,
  forbid_watermarking: false,
  max_cost_usd_per_task: null,
  max_latency_s: null,
};

const CONFIG: RoutingConfig = {
  goal: "quality",
  guardrails: GUARDRAILS,
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
    guardrails: GUARDRAILS,
  },
  routes: [
    {
      id: "route-sensitive-ip",
      work_type: null,
      sensitivity: "sensitive-ip",
      goal: "privacy",
      target: "auto",
      guardrails: {
        ...GUARDRAILS,
        forbid_training_retention: true,
        forbid_watermarking: true,
      },
    },
    {
      id: "route-locked",
      work_type: "coding",
      sensitivity: "default",
      goal: "quality",
      target: {
        provider_key: "anthropic",
        model: "anthropic/claude-sonnet-4-5",
      },
      guardrails: GUARDRAILS,
    },
    {
      id: "route-default",
      work_type: null,
      sensitivity: null,
      goal: "quality",
      target: "auto",
      guardrails: GUARDRAILS,
    },
  ],
  work_types: [
    { id: "coding", name: "Coding", description: "Implement features." },
  ],
  sensitivities: [
    { id: "default", name: "Default", description: "Ordinary work." },
    {
      id: "sensitive-ip",
      name: "Sensitive IP",
      description: "Proprietary code.",
    },
  ],
  imported_project_path: null,
};

describe("RoutesTable", () => {
  it("renders chips, auto preview, and locked pill", () => {
    renderWithProviders(
      <RoutesTable
        config={CONFIG}
        previews={{ "route-sensitive-ip": "openhands/openhands/glm-5.2" }}
        onChange={vi.fn()}
      />,
    );
    expect(
      screen.getByTestId("routing-chip-sensitivity-route-sensitive-ip"),
    ).toHaveTextContent("sensitive-ip");
    expect(
      screen.getByTestId("routing-target-auto-route-sensitive-ip"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("routing-locked-route-locked")).toHaveTextContent(
      I18nKey.ROUTING$LOCKED,
    );
    expect(
      screen.getByTestId("routing-chip-work-route-locked"),
    ).toHaveTextContent("coding");
  });
});

describe("RouterModelChooser", () => {
  const response: RoutingRouterModelResponse = {
    config: CONFIG.router_model,
    resolved: {
      preset: "cheapest",
      provider_key: "openhands",
      model: "openhands/glm-5.2",
      goal: "cost",
      guardrails: GUARDRAILS,
      tradeoffs: {
        cost_per_1k: 0.0012,
        latency_s_p90: 18,
        retention: "none",
        watermark: "none",
        classification_accuracy_proxy: 0.76,
        offline_capable: false,
        verified: false,
      },
      pool: [
        {
          id: "openhands/glm-5.2",
          provider_key: "openhands",
          cost_per_1k: 0.0012,
          latency_s_p90: 18,
          retention: "none",
          watermark: "none",
          classification_accuracy_proxy: 0.76,
          offline_capable: false,
          verified: false,
        },
        {
          id: "ollama/qwen3-coder:16b",
          provider_key: "ollama",
          cost_per_1k: 0,
          latency_s_p90: 80,
          retention: "none",
          watermark: "none",
          classification_accuracy_proxy: 0.5,
          offline_capable: true,
          verified: false,
        },
        {
          id: "ollama/llama3.2:3b",
          provider_key: "ollama",
          cost_per_1k: 0,
          latency_s_p90: 18,
          retention: "none",
          watermark: "none",
          classification_accuracy_proxy: 0.3,
          offline_capable: true,
          verified: false,
        },
      ],
    },
  };

  const runtimes: RoutingLocalRuntimes = {
    runtimes: {
      ollama: { alive: true, models: ["qwen3-coder:16b"], error: null },
    },
  };

  it("renders trade-offs and filters Local to installed models", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    renderWithProviders(
      <RouterModelChooser
        routerModel={{
          ...response,
          config: { ...response.config, preset: "local" },
        }}
        localRuntimes={runtimes}
        onSelectPreset={onSelect}
      />,
    );
    expect(
      screen.getByTestId("routing-preset-cheapest-cost"),
    ).toHaveTextContent("0.0012");
    expect(
      screen.getByTestId("routing-pool-ollama/qwen3-coder:16b"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("routing-pool-ollama/llama3.2:3b"),
    ).not.toBeInTheDocument();
    await user.click(screen.getByTestId("routing-preset-local"));
    expect(onSelect).toHaveBeenCalledWith("local");
  });
});

describe("TaxonomyEditor", () => {
  it("previews the classifier prompt and supports CRUD", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithProviders(
      <TaxonomyEditor
        taxonomy={{
          work_types: CONFIG.work_types,
          sensitivities: CONFIG.sensitivities,
          classifier_prompt: "",
        }}
        onChange={onChange}
        onReset={vi.fn()}
      />,
    );
    expect(screen.getByTestId("routing-classifier-prompt")).toHaveTextContent(
      "coding",
    );
    expect(screen.getByTestId("routing-classifier-prompt")).toHaveTextContent(
      "Implement features.",
    );
    await user.click(screen.getByTestId("routing-work-types-add"));
    expect(onChange).toHaveBeenCalled();
    await user.click(screen.getByTestId("routing-work-types-delete-coding"));
    expect(onChange).toHaveBeenCalled();
  });
});

describe("BenchmarkSources", () => {
  const sources: RoutingSourcesResponse = {
    sources: [
      {
        id: "swebench",
        last_success: "2026-09-06T00:00:00Z",
        last_error: "network down",
        stale: true,
      },
    ],
  };

  it("shows status and ingest-now", async () => {
    const user = userEvent.setup();
    const onIngest = vi.fn();
    renderWithProviders(
      <BenchmarkSources sources={sources} onIngest={onIngest} />,
    );
    expect(screen.getByTestId("routing-source-swebench")).toBeInTheDocument();
    expect(
      screen.getByTestId("routing-source-error-swebench"),
    ).toHaveTextContent("network down");
    expect(
      screen.getByTestId("routing-source-stale-swebench"),
    ).toBeInTheDocument();
    await user.click(screen.getByTestId("routing-ingest-now"));
    expect(onIngest).toHaveBeenCalled();
  });
});

describe("ModelRegistryBrowser", () => {
  it("renders badges and provenance tooltip", () => {
    const registry: RoutingRegistrySnapshot = {
      version: "v1",
      last_updated: "2026-09-06T00:00:00Z",
      privacy_last_updated: "2026-09-06T00:00:00Z",
      stale: false,
      privacy_stale: true,
      models: [
        {
          id: "anthropic/claude-sonnet-4-5",
          provider_key: "anthropic",
          benchmarks: {
            coding: {
              score: 0.8,
              provenance: {
                source: "swebench",
                fetched_at: "2026-09-06T00:00:00Z",
                version: "swebench",
              },
            },
          },
          cost_per_1k: 0.006,
          latency_s_p90: 16,
          local: false,
          runtime: null,
          source_url: "https://www.swebench.com/verified",
          verified: true,
          notes: null,
          retention: "none",
          watermark: "none",
          reachable: true,
        },
        {
          id: "google/gemini-2.5-pro",
          provider_key: "gemini",
          benchmarks: {},
          cost_per_1k: 0.0025,
          latency_s_p90: 14,
          local: false,
          runtime: null,
          source_url: null,
          verified: false,
          notes: null,
          retention: "opt-out",
          watermark: "always",
          reachable: false,
        },
      ],
      sources: {},
    };
    renderWithProviders(<ModelRegistryBrowser registry={registry} />);
    expect(screen.getByTestId("routing-privacy-stale")).toBeInTheDocument();
    expect(
      screen.getByTestId("routing-model-verified-anthropic/claude-sonnet-4-5"),
    ).toHaveTextContent(I18nKey.ROUTING$VERIFIED);
    expect(
      screen.getByTestId("routing-bench-coding-anthropic/claude-sonnet-4-5"),
    ).toHaveAttribute("title", "swebench");
    expect(
      screen.getByTestId("routing-model-reachable-google/gemini-2.5-pro"),
    ).toHaveTextContent(I18nKey.ROUTING$UNREACHABLE);
  });
});

describe("HowAutoRoutes", () => {
  it("lists precedence", () => {
    renderWithProviders(<HowAutoRoutes />);
    expect(
      screen.getByTestId("routing-precedence-specific"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("routing-precedence-sensitivity"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("routing-precedence-default"),
    ).toBeInTheDocument();
  });
});

describe("DecisionTraceDrawer", () => {
  it("renders the full trace", () => {
    const trace: RoutingTrace = {
      task_text: "implement a parser",
      classification: {
        work_type: "coding",
        sensitivity: "default",
        complexity: "medium",
        confidence: 0.8,
        reason: "explicit",
        classifier_version: "v1-taxonomy",
      },
      classifier_version: "v1-taxonomy",
      route_id: "route-default",
      filters: [{ id: "google/gemini-2.5-pro", reason: "not connected" }],
      ranked: [
        {
          id: "openhands/glm-5.2",
          provider_key: "openhands",
          score: 0.78,
          score_source: "benchmark:coding",
        },
      ],
      chosen: {
        provider_key: "openhands",
        model: "openhands/glm-5.2",
        score: 0.78,
        rule_id: "route-default",
        registry_version: "v1",
        target: "auto",
        usable: true,
        goal: "quality",
      },
      reason: "Auto-picked openhands/glm-5.2",
    };
    renderWithProviders(
      <DecisionTraceDrawer trace={trace} onClose={vi.fn()} />,
    );
    expect(screen.getByTestId("routing-trace-task")).toHaveTextContent(
      "implement a parser",
    );
    expect(
      screen.getByTestId("routing-trace-classification"),
    ).toHaveTextContent("coding");
    expect(screen.getByTestId("routing-trace-filters")).toHaveTextContent(
      "not connected",
    );
    expect(screen.getByTestId("routing-trace-ranked")).toHaveTextContent(
      "benchmark:coding",
    );
    expect(screen.getByTestId("routing-trace-reason")).toHaveTextContent(
      "Auto-picked openhands/glm-5.2",
    );
  });
});

describe("GoalGuardrails", () => {
  it("fires config updates from toggles", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithProviders(<GoalGuardrails config={CONFIG} onChange={onChange} />);
    await user.click(screen.getByLabelText(I18nKey.ROUTING$MODE));
    expect(onChange).toHaveBeenCalledWith({ mode: "strict" });
  });
});

describe("DryRunConsole", () => {
  it("resolves and renders the reason", async () => {
    const user = userEvent.setup();
    const onResolve = vi.fn();
    renderWithProviders(
      <DryRunConsole
        onResolve={onResolve}
        result={{
          decision: {
            provider_key: "openhands",
            model: "openhands/glm-5.2",
            score: 0.78,
            rule_id: "route-default",
            registry_version: "v1",
            target: "auto",
            usable: true,
            goal: "quality",
          },
          trace: {
            task_text: "fix the button",
            classification: {
              work_type: "ux",
              sensitivity: "default",
              complexity: "low",
              confidence: 1,
              reason: "",
            },
            classifier_version: "v1-taxonomy",
            route_id: "route-ux",
            filters: [],
            ranked: [],
            chosen: {
              provider_key: "openhands",
              model: "openhands/glm-5.2",
              score: 0.78,
              rule_id: "route-ux",
              registry_version: "v1",
              target: "auto",
              usable: true,
              goal: "quality",
            },
            reason: "because quality",
          },
        }}
      />,
    );
    await user.type(
      screen.getByTestId("routing-dry-run-input"),
      "fix the button",
    );
    await user.click(screen.getByTestId("routing-dry-run-resolve"));
    expect(onResolve).toHaveBeenCalledWith("fix the button");
    expect(screen.getByTestId("routing-dry-run-reason")).toHaveTextContent(
      "because quality",
    );
  });
});

describe("SwitchHistory", () => {
  const items: RoutingAuditItem[] = [
    {
      id: "switch-1",
      created_at: "2026-09-06T00:00:00Z",
      kind: "switch",
      card_id: "card-1",
      run_id: "run-1",
      payload: {
        from: {
          provider_key: "anthropic",
          model: "anthropic/claude-opus-4-6",
        },
        to: {
          provider_key: "opencode",
          model: "opencode/anthropic/claude-sonnet-4-6",
        },
        reason: "struggle",
      },
    },
    {
      id: "resolve-1",
      created_at: "2026-09-06T00:00:00Z",
      kind: "resolve",
      card_id: "card-1",
      run_id: "run-1",
      payload: {
        decision: {
          provider_key: "anthropic",
          model: "anthropic/claude-opus-4-6",
          score: 0.86,
          rule_id: "route-default",
          registry_version: "v1",
          target: "auto",
          usable: true,
          goal: "quality",
        },
        trace: {
          task_text: "implement a parser",
          classification: {
            work_type: "coding",
            sensitivity: "default",
            complexity: "medium",
            confidence: 1,
            reason: "",
          },
          classifier_version: "v1-taxonomy",
          route_id: "route-default",
          filters: [],
          ranked: [
            {
              id: "anthropic/claude-opus-4-6",
              provider_key: "anthropic",
              score: 0.86,
              score_source: "benchmark:coding",
              cost_per_1k: 0.03,
            },
          ],
          chosen: {
            provider_key: "anthropic",
            model: "anthropic/claude-opus-4-6",
            score: 0.86,
            rule_id: "route-default",
            registry_version: "v1",
            target: "auto",
            usable: true,
            goal: "quality",
          },
          reason: "auto",
        },
      },
    },
  ];

  it("renders switch-from/to and cost-per-task", () => {
    renderWithProviders(<SwitchHistory items={items} />);
    expect(screen.getByTestId("routing-switch-history")).toBeInTheDocument();
    expect(screen.getByTestId("routing-switch-from-switch-1")).toHaveTextContent(
      "anthropic/claude-opus-4-6",
    );
    expect(screen.getByTestId("routing-switch-to-switch-1")).toHaveTextContent(
      "opencode/anthropic/claude-sonnet-4-6",
    );
    expect(
      screen.getByTestId("routing-switch-reason-switch-1"),
    ).toHaveTextContent(I18nKey.ROUTING$STRUGGLE);
    expect(screen.getByTestId("routing-audit-cost-resolve-1")).toHaveTextContent(
      "$0.03",
    );
  });
});

describe("ingest failure toast", () => {
  it("toasts when ingest reports a failed source", async () => {
    const user = userEvent.setup();
    const toast = vi
      .spyOn(toastHandlers, "displayErrorToast")
      .mockImplementation(() => undefined);
    vi.spyOn(RoutingService, "getConfig").mockResolvedValue(CONFIG);
    vi.spyOn(RoutingService, "getTaxonomy").mockResolvedValue({
      work_types: CONFIG.work_types,
      sensitivities: CONFIG.sensitivities,
      classifier_prompt: "coding",
    });
    vi.spyOn(RoutingService, "getRegistry").mockResolvedValue({
      version: "v1",
      last_updated: "2026-09-06T00:00:00Z",
      privacy_last_updated: "2026-09-06T00:00:00Z",
      stale: false,
      privacy_stale: false,
      models: [],
      sources: {},
    });
    vi.spyOn(RoutingService, "getSources").mockResolvedValue({
      sources: [
        { id: "swebench", last_success: null, last_error: null, stale: false },
      ],
    });
    vi.spyOn(RoutingService, "getRouterModel").mockResolvedValue({
      config: CONFIG.router_model,
      resolved: {
        preset: "cheapest",
        provider_key: "openhands",
        model: "openhands/glm-5.2",
        goal: "cost",
        guardrails: GUARDRAILS,
        tradeoffs: {
          cost_per_1k: 0.0012,
          latency_s_p90: 18,
          retention: "none",
          watermark: "none",
          classification_accuracy_proxy: 0.7,
          offline_capable: false,
          verified: false,
        },
        pool: [],
      },
    });
    vi.spyOn(RoutingService, "getLocalRuntimes").mockResolvedValue({
      runtimes: {
        ollama: { alive: true, models: ["qwen3-coder:16b"], error: null },
      },
    });
    vi.spyOn(RoutingService, "getAudit").mockResolvedValue({
      items: [],
      total: 0,
      limit: 50,
      offset: 0,
    });
    vi.spyOn(RoutingService, "resolve").mockResolvedValue({
      decision: {
        provider_key: "openhands",
        model: "openhands/glm-5.2",
        score: 0.78,
        rule_id: "route-default",
        registry_version: "v1",
        target: "auto",
        usable: true,
        goal: "quality",
      },
      trace: {
        task_text: "x",
        classification: {
          work_type: "coding",
          sensitivity: "default",
          complexity: "low",
          confidence: 1,
          reason: "",
        },
        classifier_version: "v1-taxonomy",
        route_id: "route-default",
        filters: [],
        ranked: [],
        chosen: {
          provider_key: "openhands",
          model: "openhands/glm-5.2",
          score: 0.78,
          rule_id: "route-default",
          registry_version: "v1",
          target: "auto",
          usable: true,
          goal: "quality",
        },
        reason: "ok",
      },
    });
    vi.spyOn(RoutingService, "ingestBenchmarks").mockResolvedValue({
      sources: { swebench: { ok: false, error: "network down" } },
      unmapped: [],
    });

    renderWithProviders(<RoutingPage />);
    await user.click(await screen.findByTestId("routing-ingest-now"));
    await waitFor(() => {
      expect(toast).toHaveBeenCalled();
    });
  });
});
