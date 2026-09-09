import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import {
  STANDARDS_ACTION_BLOCK,
  STANDARDS_ACTION_WARN,
  STANDARDS_SEVERITY_WARNING,
  STANDARDS_SOURCE_BUILTIN,
} from "#/api/standards-service/standards-constants";
import StandardsService from "#/api/standards-service/standards-service.api";
import type {
  StandardsAuditPage,
  StandardsConfig,
  StandardsPluginInfo,
  StandardsRunResult,
} from "#/api/standards-service/standards-types";
import { StandardsAuditLog } from "#/components/features/standards/standards-audit-log";
import { StandardsConsole } from "#/components/features/standards/standards-console";
import { StandardsEnforcement } from "#/components/features/standards/standards-enforcement";
import { StandardsPage } from "#/components/features/standards/standards-page";
import { StandardsPluginGallery } from "#/components/features/standards/standards-plugin-gallery";
import { I18nKey } from "#/i18n/declaration";

const PLUGIN: StandardsPluginInfo = {
  name: "demo",
  display_name: "Demo (tab indentation)",
  description: "Flags tab indentation",
  version: "1.0.0",
  source: STANDARDS_SOURCE_BUILTIN,
  enabled: true,
  action: STANDARDS_ACTION_WARN,
  severity_default: STANDARDS_SEVERITY_WARNING,
};

const CONFIG: StandardsConfig = {
  enabled: true,
  enforcement: { prompt: true, automated: true, gates: true },
  plugins: [{ name: "demo", enabled: true, action: STANDARDS_ACTION_WARN }],
  project_yaml_source: true,
};

const RUN: StandardsRunResult = {
  run_id: "run-1",
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
};

const AUDIT: StandardsAuditPage = {
  items: [
    {
      id: "audit-1",
      run_id: "run-1",
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
  ],
  limit: 50,
  next_before_id: "audit-1",
};

describe("StandardsPluginGallery", () => {
  it("toggles a plugin", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();
    renderWithProviders(
      <StandardsPluginGallery plugins={[PLUGIN]} onToggle={onToggle} />,
    );
    expect(screen.getByTestId("standards-plugin-card-demo")).toHaveTextContent(
      "Demo (tab indentation)",
    );
    await user.click(screen.getByRole("switch"));
    expect(onToggle).toHaveBeenCalledWith(PLUGIN);
  });
});

describe("StandardsEnforcement", () => {
  it("persists layer toggles and per-plugin action", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithProviders(
      <StandardsEnforcement
        config={CONFIG}
        plugins={[PLUGIN]}
        onChange={onChange}
      />,
    );
    expect(
      screen.getByTestId("standards-project-yaml-notice"),
    ).toBeInTheDocument();
    await user.click(screen.getByLabelText(I18nKey.STANDARDS$PROMPT));
    expect(onChange).toHaveBeenCalledWith({
      enforcement: { prompt: false, automated: true, gates: true },
    });
    await user.selectOptions(
      screen.getByTestId("standards-action-demo"),
      STANDARDS_ACTION_BLOCK,
    );
    expect(onChange).toHaveBeenCalledWith({
      plugins: [
        { name: "demo", enabled: true, action: STANDARDS_ACTION_BLOCK },
      ],
    });
  });
});

describe("StandardsConsole", () => {
  it("runs checks and shows summary tiles plus violations", async () => {
    const user = userEvent.setup();
    const onRun = vi.fn();
    renderWithProviders(
      <StandardsConsole
        root="workspace/project"
        onRootChange={vi.fn()}
        onRun={onRun}
        result={RUN}
      />,
    );
    expect(screen.getByTestId("standards-run-id")).toHaveTextContent("run-1");
    expect(screen.getByTestId("standards-files-scanned")).toHaveTextContent(
      "2",
    );
    expect(screen.getByTestId("standards-summary-warning")).toHaveTextContent(
      "1",
    );
    expect(
      screen.getByTestId("standards-violation-DEMO-TAB-INDENT"),
    ).toHaveTextContent("src/app.py:2");
    await user.click(screen.getByTestId("standards-run"));
    expect(onRun).toHaveBeenCalled();
  });
});

describe("StandardsAuditLog", () => {
  it("filters and pages older rows", async () => {
    const user = userEvent.setup();
    const onPluginChange = vi.fn();
    const onLoadOlder = vi.fn();
    renderWithProviders(
      <StandardsAuditLog
        page={AUDIT}
        plugin=""
        severity=""
        file=""
        onPluginChange={onPluginChange}
        onSeverityChange={vi.fn()}
        onFileChange={vi.fn()}
        onLoadOlder={onLoadOlder}
      />,
    );
    expect(screen.getByTestId("standards-audit-row-audit-1")).toHaveTextContent(
      "demo",
    );
    await user.type(screen.getByTestId("standards-audit-plugin"), "demo");
    expect(onPluginChange).toHaveBeenCalled();
    await user.click(screen.getByTestId("standards-audit-load-older"));
    expect(onLoadOlder).toHaveBeenCalled();
  });
});

describe("StandardsPage", () => {
  beforeEach(() => {
    vi.spyOn(StandardsService, "listPlugins").mockResolvedValue({
      plugins: [PLUGIN],
      load_errors: [],
    });
    vi.spyOn(StandardsService, "getConfig").mockResolvedValue(CONFIG);
    vi.spyOn(StandardsService, "putConfig").mockResolvedValue({
      ...CONFIG,
      plugins: [
        { name: "demo", enabled: false, action: STANDARDS_ACTION_WARN },
      ],
    });
    vi.spyOn(StandardsService, "run").mockResolvedValue(RUN);
    vi.spyOn(StandardsService, "getAudit").mockResolvedValue(AUDIT);
  });

  it("loads gallery, runs checks, and writes config", async () => {
    const user = userEvent.setup();
    renderWithProviders(<StandardsPage />);
    expect(
      await screen.findByTestId("standards-plugin-card-demo"),
    ).toBeInTheDocument();
    await user.click(screen.getByTestId("standards-run"));
    await waitFor(() => {
      expect(StandardsService.run).toHaveBeenCalled();
    });
    expect(
      await screen.findByTestId("standards-violations-table"),
    ).toBeInTheDocument();
    await user.click(screen.getAllByRole("switch")[0]);
    await waitFor(() => {
      expect(StandardsService.putConfig).toHaveBeenCalled();
    });
  });
});
