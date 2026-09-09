import { http, HttpResponse } from "msw";
import {
  DEFAULT_STANDARDS_AUDIT_LIMIT,
  STANDARDS_ACTION_WARN,
  STANDARDS_AUDIT_PATH,
  STANDARDS_CONFIG_PATH,
  STANDARDS_PLUGINS_PATH,
  STANDARDS_RUN_PATH,
  STANDARDS_SEVERITY_WARNING,
  STANDARDS_SOURCE_BUILTIN,
} from "#/api/standards-service/standards-constants";
import type {
  StandardsAuditItem,
  StandardsConfig,
  StandardsPluginInfo,
  StandardsRunResult,
} from "#/api/standards-service/standards-types";

const DEFAULT_PLUGIN: StandardsPluginInfo = {
  name: "demo",
  display_name: "Demo (tab indentation)",
  description: "Flags lines that start with a tab character.",
  version: "1.0.0",
  source: STANDARDS_SOURCE_BUILTIN,
  enabled: true,
  action: STANDARDS_ACTION_WARN,
  severity_default: STANDARDS_SEVERITY_WARNING,
};

const DEFAULT_CONFIG: StandardsConfig = {
  enabled: true,
  enforcement: { prompt: true, automated: true, gates: true },
  plugins: [{ name: "demo", enabled: true, action: STANDARDS_ACTION_WARN }],
  project_yaml_source: false,
};

let plugins: StandardsPluginInfo[] = [{ ...DEFAULT_PLUGIN }];
let config: StandardsConfig = structuredClone(DEFAULT_CONFIG);
let auditItems: StandardsAuditItem[] = [];
let runCounter = 1;

export function resetStandardsMockData() {
  plugins = [{ ...DEFAULT_PLUGIN }];
  config = structuredClone(DEFAULT_CONFIG);
  auditItems = [];
  runCounter = 1;
}

export const STANDARDS_HANDLERS = [
  http.get(`*${STANDARDS_PLUGINS_PATH}`, () =>
    HttpResponse.json({ plugins, load_errors: [] }),
  ),
  http.get(`*${STANDARDS_CONFIG_PATH}`, () => HttpResponse.json(config)),
  http.put(`*${STANDARDS_CONFIG_PATH}`, async ({ request }) => {
    const body = (await request
      .json()
      .catch(() => ({}))) as Partial<StandardsConfig>;
    if (typeof body.enabled === "boolean") config.enabled = body.enabled;
    if (body.enforcement) {
      config.enforcement = { ...config.enforcement, ...body.enforcement };
    }
    if (Array.isArray(body.plugins)) {
      config.plugins = body.plugins;
      plugins = plugins.map((plugin) => {
        const match = body.plugins?.find((item) => item.name === plugin.name);
        return match
          ? { ...plugin, enabled: match.enabled, action: match.action }
          : plugin;
      });
    }
    return HttpResponse.json(config);
  }),
  http.post(`*${STANDARDS_RUN_PATH}`, async ({ request }) => {
    const body = (await request.json().catch(() => ({}))) as { root?: string };
    const runId = `run-${runCounter}`;
    runCounter += 1;
    const result: StandardsRunResult = {
      run_id: runId,
      summary: {
        files_scanned: 2,
        violation_count: 1,
        info: 0,
        warning: 1,
        error: 0,
      },
      violations: [
        {
          plugin_name: "demo",
          rule_id: "DEMO-TAB-INDENT",
          severity: STANDARDS_SEVERITY_WARNING,
          file: "src/app.py",
          line: 2,
          message: "Line uses tab indentation",
          remediation: "Replace leading tabs with spaces",
          action: STANDARDS_ACTION_WARN,
        },
      ],
      duration_ms: 12,
      status: "passed",
      worktree: body.root || "/workspace/project",
    };
    auditItems = [
      {
        id: `audit-${runId}`,
        run_id: runId,
        created_at: "2026-01-01T00:00:00+00:00",
        plugin_name: "demo",
        rule_id: "DEMO-TAB-INDENT",
        severity: STANDARDS_SEVERITY_WARNING,
        file: "src/app.py",
        line: 2,
        message: "Line uses tab indentation",
        remediation: "Replace leading tabs with spaces",
        action: STANDARDS_ACTION_WARN,
      },
      ...auditItems,
    ];
    return HttpResponse.json(result);
  }),
  http.get(`*${STANDARDS_AUDIT_PATH}`, ({ request }) => {
    const url = new URL(request.url);
    const plugin = url.searchParams.get("plugin");
    const severity = url.searchParams.get("severity");
    const file = url.searchParams.get("file");
    const runId = url.searchParams.get("run_id");
    const beforeId = url.searchParams.get("before_id");
    const limit = Number(
      url.searchParams.get("limit") || DEFAULT_STANDARDS_AUDIT_LIMIT,
    );
    let items = auditItems;
    if (plugin) items = items.filter((item) => item.plugin_name === plugin);
    if (severity) items = items.filter((item) => item.severity === severity);
    if (file) items = items.filter((item) => item.file.includes(file));
    if (runId) items = items.filter((item) => item.run_id === runId);
    if (beforeId) {
      const index = items.findIndex((item) => item.id === beforeId);
      if (index >= 0) items = items.slice(index + 1);
    }
    const page = items.slice(0, limit);
    return HttpResponse.json({
      items: page,
      limit,
      next_before_id: items.length > limit ? page[page.length - 1]?.id : null,
    });
  }),
];
