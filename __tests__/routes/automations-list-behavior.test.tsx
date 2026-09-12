import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { I18nKey } from "#/i18n/declaration";
import type { Automation, AutomationSpec } from "#/types/automation";

const mocks = vi.hoisted(() => ({
  healthState: {
    data: { status: "ok" } as { status: string } | undefined,
    isLoading: false,
    refetch: vi.fn(),
  },
  automationsState: {
    data: { automations: [], total: 0 } as
      | { automations: Automation[]; total: number }
      | undefined,
    isLoading: false,
    isError: false,
    refetch: vi.fn(),
  },
  dispatchState: {
    isPending: false,
    variables: undefined as string | undefined,
  },
  importState: { isPending: false },
  backendKind: "local" as "local" | "cloud",
  canManage: true,
  navigate: vi.fn(),
  useAutomations: vi.fn(),
  toggle: vi.fn(),
  remove: vi.fn(),
  dispatch: vi.fn(),
  importAutomation: vi.fn(),
  trackEnabled: vi.fn(),
  trackExported: vi.fn(),
  useTranslation: vi.fn(),
  translate: vi.fn((key: string) => key),
  successToast: vi.fn(),
  successToastWithLink: vi.fn(),
  errorToast: vi.fn(),
  serializeAutomation: vi.fn(),
  parseAutomationFile: vi.fn(),
  exportFilename: vi.fn(),
  downloadBlob: vi.fn(),
}));

vi.mock("react-i18next", () => ({
  useTranslation: (namespace: string) => {
    mocks.useTranslation(namespace);
    return { t: mocks.translate };
  },
}));

// Pin the published data source to a manifest that declares no dashboard or
// sub-page surface so the route renders in its plain-list form — the behavior
// this suite exercises — independent of whatever the pinned package ships.
vi.mock("#/manifests/manifest-sources", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/manifests/manifest-sources")>();
  const { createInterfaceManifest } = await import(
    "../manifests/manifest-test-data"
  );
  return {
    ...actual,
    AUTOMATION_INTERFACE_CANDIDATE: createInterfaceManifest(),
  };
});

vi.mock("#/hooks/use-automation-permissions", () => ({
  useAutomationPermissions: () => ({ canManage: mocks.canManage }),
}));

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({ navigate: mocks.navigate }),
}));

vi.mock("#/hooks/query/use-automation-run-summaries", () => ({
  useAutomationRunSummaries: () => ({}),
}));

vi.mock("#/hooks/query/use-automation-health", () => ({
  useAutomationHealth: () => mocks.healthState,
}));

vi.mock("#/hooks/query/use-automations", () => ({
  useAutomations: (options: unknown) => {
    mocks.useAutomations(options);
    return mocks.automationsState;
  },
  useToggleAutomation: () => ({ mutate: mocks.toggle }),
  useDeleteAutomation: () => ({ mutate: mocks.remove }),
  useDispatchAutomation: () => ({
    mutate: mocks.dispatch,
    ...mocks.dispatchState,
  }),
  useImportAutomation: () => ({
    mutate: mocks.importAutomation,
    ...mocks.importState,
  }),
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "backend-1", kind: mocks.backendKind },
  }),
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackPrebuiltAutomationEnabled: mocks.trackEnabled,
    trackAutomationExported: mocks.trackExported,
  }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: mocks.successToast,
  displaySuccessToastWithLink: mocks.successToastWithLink,
  displayErrorToast: mocks.errorToast,
}));

vi.mock("#/utils/automation-export", () => ({
  serializeAutomation: mocks.serializeAutomation,
  parseAutomationFile: mocks.parseAutomationFile,
  getAutomationExportFilename: mocks.exportFilename,
}));

vi.mock("#/utils/utils", () => ({
  downloadBlob: mocks.downloadBlob,
  cn: (...classes: unknown[]) => classes.filter(Boolean).join(" "),
}));

vi.mock("#/components/features/automations/search-input", () => ({
  SearchInput: ({
    value,
    onChange,
  }: {
    value: string;
    onChange: (value: string) => void;
  }) => (
    <input
      aria-label="automation-search"
      value={value}
      onChange={(event) => onChange(event.target.value)}
    />
  ),
}));

interface AutomationGroupProps {
  title: string;
  count: number;
  automations: Automation[];
  view: string;
  runPendingId: string | null;
  onToggle: (id: string, enabled: boolean) => void;
  onRunNow: (id: string) => void;
  onDelete: (id: string) => void;
  onExport: (automation: Automation) => void;
  onEdit?: (id: string) => void;
}

vi.mock("#/components/features/automations/automation-group", () => ({
  AutomationGroup: ({
    title,
    count,
    automations,
    view,
    runPendingId,
    onToggle,
    onRunNow,
    onDelete,
    onExport,
    onEdit,
  }: AutomationGroupProps) => (
    <section aria-label={title} data-count={count} data-view={view}>
      <output data-testid={`pending-${title}`}>{runPendingId ?? "none"}</output>
      <button type="button" onClick={() => onToggle("missing-id", false)}>
        stale-toggle-{title}
      </button>
      <button type="button" onClick={() => onDelete("missing-id")}>
        stale-delete-{title}
      </button>
      {onEdit && (
        <button type="button" onClick={() => onEdit("missing-id")}>
          stale-edit-{title}
        </button>
      )}
      {automations.map((automation) => (
        <article
          key={automation.id}
          data-testid={`automation-${automation.id}`}
        >
          <span>{automation.name}</span>
          <button
            type="button"
            onClick={() => onToggle(automation.id, automation.enabled)}
          >
            toggle-{automation.id}
          </button>
          <button type="button" onClick={() => onRunNow(automation.id)}>
            run-{automation.id}
          </button>
          <button type="button" onClick={() => onDelete(automation.id)}>
            delete-{automation.id}
          </button>
          <button type="button" onClick={() => onExport(automation)}>
            export-{automation.id}
          </button>
          {onEdit && (
            <button type="button" onClick={() => onEdit(automation.id)}>
              edit-{automation.id}
            </button>
          )}
        </article>
      ))}
    </section>
  ),
}));

vi.mock("#/components/features/automations/automation-view-toggle", () => ({
  AutomationViewToggle: ({
    view,
    onChange,
    disabled,
  }: {
    view: string;
    onChange: (view: "grid" | "list") => void;
    disabled: boolean;
  }) => (
    <button
      type="button"
      disabled={disabled}
      data-testid="view-toggle"
      onClick={() => onChange(view === "grid" ? "list" : "grid")}
    >
      {view}
    </button>
  ),
}));

vi.mock("#/components/features/automations/automation-card-skeleton", () => ({
  AutomationCardSkeleton: () => <div data-testid="automation-skeleton" />,
}));

vi.mock("#/components/features/automations/empty-state", () => ({
  EmptyState: () => <div data-testid="empty-state" />,
}));

vi.mock("#/components/features/automations/error-state", () => ({
  ErrorState: ({ onRetry }: { onRetry: () => void }) => (
    <button type="button" onClick={onRetry}>
      retry-list
    </button>
  ),
}));

vi.mock("#/components/features/automations/backend-not-configured", () => ({
  BackendNotConfigured: ({ onRetry }: { onRetry: () => void }) => (
    <button type="button" onClick={onRetry}>
      retry-health
    </button>
  ),
}));

vi.mock("#/components/features/automations/delete-confirmation-modal", () => ({
  DeleteConfirmationModal: ({
    automationName,
    isOpen,
    onConfirm,
    onCancel,
  }: {
    automationName: string;
    isOpen: boolean;
    onConfirm: () => void;
    onCancel: () => void;
  }) =>
    isOpen ? (
      <div data-testid="delete-modal">
        <span>{automationName}</span>
        <button type="button" onClick={onConfirm}>
          confirm-delete
        </button>
        <button type="button" onClick={onCancel}>
          cancel-delete
        </button>
      </div>
    ) : null,
}));

vi.mock(
  "#/components/features/automations/detail/edit-automation-modal",
  () => ({
    EditAutomationModal: ({
      automation,
      isOpen,
      onClose,
    }: {
      automation: Automation;
      isOpen: boolean;
      onClose: () => void;
    }) =>
      isOpen ? (
        <div data-testid="edit-modal">
          <span>{automation.name}</span>
          <button type="button" onClick={onClose}>
            close-edit
          </button>
        </div>
      ) : null,
  }),
);

vi.mock("#/components/features/automations/add-automation-modal", () => ({
  AddAutomationModal: ({
    isOpen,
    onClose,
  }: {
    isOpen: boolean;
    onClose: () => void;
  }) =>
    isOpen ? (
      <div data-testid="add-modal">
        <button type="button" onClick={onClose}>
          close-add
        </button>
      </div>
    ) : null,
}));

vi.mock("#/components/features/automations/import-automation-modal", () => ({
  ImportAutomationModal: ({
    isOpen,
    spec,
    isImporting,
    onClose,
    onImport,
    onFile,
  }: {
    isOpen: boolean;
    spec: AutomationSpec | null;
    isImporting: boolean;
    onClose: () => void;
    onImport: () => void;
    onFile: (file: File) => void;
  }) =>
    isOpen ? (
      <div data-testid="import-modal" data-importing={isImporting}>
        <span>{spec?.name}</span>
        <input
          type="file"
          data-testid="automations-import-file"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onFile(file);
          }}
        />
        <button type="button" onClick={onImport}>
          confirm-import
        </button>
        <button type="button" onClick={onClose}>
          close-import
        </button>
      </div>
    ) : null,
}));

vi.mock(
  "#/components/features/automations/recommended-automations-launcher",
  () => ({
    RecommendedAutomationsLauncher: ({ query }: { query: string }) => (
      <output data-testid="recommended-query">{query}</output>
    ),
  }),
);

vi.mock("#/components/features/settings/brand-button", () => ({
  BrandButton: ({
    children,
    onClick,
    testId,
  }: {
    children: React.ReactNode;
    onClick?: () => void;
    testId?: string;
  }) => (
    <button type="button" data-testid={testId} onClick={onClick}>
      {children}
    </button>
  ),
}));

vi.mock("#/components/features/automations/add-automation-menu", () => ({
  AddAutomationMenu: ({
    onAdd,
    onImport,
    isAddDisabled,
  }: {
    onAdd: () => void;
    onImport: () => void;
    isAddDisabled?: boolean;
  }) => (
    <div>
      <button
        type="button"
        data-testid="automations-add-automation"
        disabled={isAddDisabled}
        onClick={onAdd}
      >
        add-automation
      </button>
      <button
        type="button"
        data-testid="automations-import-automation"
        onClick={onImport}
      >
        import-automation
      </button>
    </div>
  ),
}));

import AutomationsList from "#/routes/automations-list";

function makeAutomation(overrides: Partial<Automation> = {}): Automation {
  return {
    id: "automation-1",
    name: "Daily digest",
    prompt: "Summarize pull requests",
    trigger: { type: "cron", schedule: "0 9 * * *" },
    enabled: true,
    repository: "openhands/agent-canvas",
    model: "fast-model",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

function makeImportSpec(
  overrides: Partial<AutomationSpec> = {},
): AutomationSpec {
  return {
    name: "Imported automation",
    prompt: "Review a change",
    trigger: { type: "cron", schedule: "0 10 * * *" },
    enabled: true,
    ...overrides,
  };
}

function renderList() {
  return render(<AutomationsList />);
}

function openImportModal() {
  fireEvent.click(screen.getByTestId("automations-import-automation"));
}

function uploadFile(text: string) {
  if (!screen.queryByTestId("automations-import-file")) {
    openImportModal();
  }
  const input = screen.getByTestId("automations-import-file");
  const file = new File([text], "automation.json", {
    type: "application/json",
  });
  Object.defineProperty(file, "text", {
    configurable: true,
    value: vi.fn().mockResolvedValue(text),
  });
  fireEvent.change(input, { target: { files: [file] } });
  return input as HTMLInputElement;
}

function readBlobText(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve(String(reader.result)));
    reader.addEventListener("error", () => reject(reader.error));
    reader.readAsText(blob);
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  window.localStorage.clear();
  mocks.healthState.data = { status: "ok" };
  mocks.healthState.isLoading = false;
  mocks.automationsState.data = { automations: [], total: 0 };
  mocks.automationsState.isLoading = false;
  mocks.automationsState.isError = false;
  mocks.dispatchState.isPending = false;
  mocks.dispatchState.variables = undefined;
  mocks.importState.isPending = false;
  mocks.backendKind = "local";
  mocks.serializeAutomation.mockReturnValue({
    version: 1,
    kind: "automation",
    spec: makeImportSpec(),
  });
  mocks.exportFilename.mockReturnValue("daily-digest.automation.json");
  mocks.parseAutomationFile.mockReturnValue(makeImportSpec());
});

describe("automations list states", () => {
  it("shows health-check placeholders and waits to enable the list query", () => {
    mocks.healthState.data = undefined;
    mocks.healthState.isLoading = true;
    mocks.automationsState.data = undefined;
    mocks.automationsState.isLoading = true;

    renderList();

    expect(screen.getAllByTestId("automation-skeleton")).toHaveLength(3);
    // The header copy comes from the admitted interface manifest.
    expect(screen.getByText("Widget automations")).toBeInTheDocument();
    expect(mocks.useAutomations).toHaveBeenCalledWith({
      limit: 50,
      offset: 0,
      enabled: false,
    });
    expect(mocks.useTranslation).toHaveBeenCalledWith("openhands");
  });

  it("offers another health check when the backend is unavailable", async () => {
    mocks.healthState.data = { status: "error" };
    const user = userEvent.setup();

    renderList();
    await user.click(screen.getByRole("button", { name: "retry-health" }));

    expect(mocks.healthState.refetch).toHaveBeenCalledOnce();
  });

  it("shows list placeholders while healthy data is loading", () => {
    mocks.automationsState.data = {
      automations: [makeAutomation()],
      total: 1,
    };
    mocks.automationsState.isLoading = true;

    renderList();

    expect(screen.getAllByTestId("automation-skeleton")).toHaveLength(3);
    expect(screen.getByTestId("recommended-query")).toHaveTextContent("");
    expect(
      screen.queryByLabelText(I18nKey.AUTOMATIONS$ACTIVE),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "retry-list" }),
    ).not.toBeInTheDocument();
    expect(mocks.useAutomations).toHaveBeenCalledWith({
      limit: 50,
      offset: 0,
      enabled: true,
    });
  });

  it("offers another list request when loading fails", async () => {
    mocks.automationsState.isError = true;
    const user = userEvent.setup();

    renderList();
    await user.click(screen.getByRole("button", { name: "retry-list" }));

    expect(mocks.automationsState.refetch).toHaveBeenCalledOnce();
    expect(screen.queryByTestId("empty-state")).not.toBeInTheDocument();
  });

  it("shows an empty state and disables view changes for an empty list", () => {
    renderList();

    expect(screen.getByTestId("empty-state")).toBeInTheDocument();
    expect(screen.getByTestId("view-toggle")).toBeDisabled();
    expect(
      screen.queryByRole("button", { name: "retry-list" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByLabelText(I18nKey.AUTOMATIONS$ACTIVE),
    ).not.toBeInTheDocument();
  });
});

describe("automations list interactions", () => {
  it("filters saved automations by name, prompt, repository, and model", async () => {
    const automations = [
      makeAutomation({ id: "by-name", name: "Needle name" }),
      makeAutomation({
        id: "by-prompt",
        name: "Second",
        prompt: "Contains needle prompt",
        enabled: false,
      }),
      makeAutomation({
        id: "by-repository",
        name: "Third",
        prompt: null,
        repository: "acme/needle-repository",
      }),
      makeAutomation({
        id: "by-model",
        name: "Fourth",
        prompt: null,
        repository: undefined,
        model: "needle-model",
      }),
      makeAutomation({
        id: "no-optional-values",
        name: "Other",
        prompt: null,
        repository: undefined,
        model: null,
      }),
    ];
    mocks.automationsState.data = { automations, total: automations.length };
    const user = userEvent.setup();

    renderList();
    expect(screen.getAllByTestId(/^automation-/)).toHaveLength(5);
    expect(
      screen.queryByRole("button", { name: "retry-list" }),
    ).not.toBeInTheDocument();

    const search = screen.getByRole("textbox", { name: "automation-search" });
    await user.type(search, "needle name");
    expect(screen.getByTestId("automation-by-name")).toBeInTheDocument();
    expect(screen.getAllByTestId(/^automation-/)).toHaveLength(1);

    await user.clear(search);
    await user.type(search, "needle prompt");
    expect(screen.getByTestId("automation-by-prompt")).toBeInTheDocument();

    await user.clear(search);
    await user.type(search, "needle-repository");
    expect(screen.getByTestId("automation-by-repository")).toBeInTheDocument();

    await user.clear(search);
    await user.type(search, "NEEDLE-MODEL");
    expect(screen.getByTestId("automation-by-model")).toBeInTheDocument();
    expect(screen.getByTestId("recommended-query")).toHaveTextContent(
      "NEEDLE-MODEL",
    );

    await user.clear(search);
    await user.type(search, "absent");
    expect(screen.queryAllByTestId(/^automation-/)).toHaveLength(0);
  });

  it("enables and disables automations while tracking only enablement", async () => {
    const enabled = makeAutomation({ id: "enabled", enabled: true });
    const disabled = makeAutomation({
      id: "disabled",
      name: "Disabled automation",
      enabled: false,
    });
    mocks.automationsState.data = {
      automations: [enabled, disabled],
      total: 2,
    };
    const user = userEvent.setup();

    renderList();
    await user.click(screen.getByRole("button", { name: "toggle-disabled" }));
    await user.click(screen.getByRole("button", { name: "toggle-enabled" }));

    expect(mocks.toggle).toHaveBeenNthCalledWith(1, {
      id: "disabled",
      enabled: true,
    });
    expect(mocks.toggle).toHaveBeenNthCalledWith(2, {
      id: "enabled",
      enabled: false,
    });
    expect(mocks.trackEnabled).toHaveBeenCalledOnce();
    expect(mocks.trackEnabled).toHaveBeenCalledWith({
      automationId: "disabled",
      automationName: "Disabled automation",
    });
  });

  it("keeps stale list actions defensive and identifies missing toggles", async () => {
    mocks.automationsState.data = {
      automations: [makeAutomation()],
      total: 1,
    };
    const user = userEvent.setup();

    renderList();
    await user.click(
      screen.getByRole("button", {
        name: `stale-toggle-${I18nKey.AUTOMATIONS$ACTIVE}`,
      }),
    );
    await user.click(
      screen.getByRole("button", {
        name: `stale-delete-${I18nKey.AUTOMATIONS$ACTIVE}`,
      }),
    );
    await user.click(
      screen.getByRole("button", {
        name: `stale-edit-${I18nKey.AUTOMATIONS$ACTIVE}`,
      }),
    );

    expect(mocks.toggle).toHaveBeenCalledWith({
      id: "missing-id",
      enabled: true,
    });
    expect(mocks.trackEnabled).toHaveBeenCalledWith({
      automationId: "missing-id",
      automationName: "missing-id",
    });
    expect(screen.queryByTestId("delete-modal")).not.toBeInTheDocument();
    expect(screen.queryByTestId("edit-modal")).not.toBeInTheDocument();
  });

  it("reports pending dispatches and confirms a successful run", async () => {
    const automation = makeAutomation();
    mocks.automationsState.data = { automations: [automation], total: 1 };
    mocks.dispatchState.isPending = true;
    mocks.dispatchState.variables = automation.id;
    mocks.dispatch.mockImplementation(
      (_id: string, options: { onSuccess: () => void }) => options.onSuccess(),
    );
    const user = userEvent.setup();

    renderList();
    expect(
      screen.getByTestId(`pending-${I18nKey.AUTOMATIONS$ACTIVE}`),
    ).toHaveTextContent(automation.id);
    expect(
      screen.getByTestId(`pending-${I18nKey.AUTOMATIONS$INACTIVE}`),
    ).toHaveTextContent(automation.id);
    await user.click(
      screen.getByRole("button", { name: `run-${automation.id}` }),
    );

    expect(mocks.dispatch).toHaveBeenCalledWith(
      automation.id,
      expect.objectContaining({
        onSuccess: expect.any(Function),
        onError: expect.any(Function),
      }),
    );
    expect(mocks.successToast).toHaveBeenCalledWith(
      I18nKey.AUTOMATIONS$RUN_NOW_SUCCESS,
    );
  });

  it("uses no pending row when a dispatch has no id yet", () => {
    mocks.automationsState.data = {
      automations: [makeAutomation()],
      total: 1,
    };
    mocks.dispatchState.isPending = true;

    renderList();

    expect(screen.getAllByText("none", { selector: "output" })).toHaveLength(2);
  });

  it.each([
    {
      label: "the backend response message",
      error: Object.assign(new Error("axios fallback"), {
        isAxiosError: true,
        response: { data: { message: "backend rejected dispatch" } },
      }),
      expected: "backend rejected dispatch",
    },
    {
      label: "the Axios message",
      error: Object.assign(new Error("network failed"), {
        isAxiosError: true,
      }),
      expected: "network failed",
    },
    {
      label: "the translated Axios fallback",
      error: Object.assign(new Error(""), { isAxiosError: true }),
      expected: I18nKey.AUTOMATIONS$RUN_NOW_ERROR,
    },
    {
      label: "a plain error message",
      error: new Error("plain failure"),
      expected: "plain failure",
    },
    {
      label: "the translated plain-error fallback",
      error: new Error(""),
      expected: I18nKey.AUTOMATIONS$RUN_NOW_ERROR,
    },
  ])("surfaces $label when a run fails", async ({ error, expected }) => {
    const automation = makeAutomation();
    mocks.automationsState.data = { automations: [automation], total: 1 };
    mocks.dispatch.mockImplementation(
      (_id: string, options: { onError: (dispatchError: unknown) => void }) =>
        options.onError(error),
    );
    const user = userEvent.setup();

    renderList();
    await user.click(
      screen.getByRole("button", { name: `run-${automation.id}` }),
    );

    expect(mocks.errorToast).toHaveBeenCalledWith(expected);
  });

  it("deletes the selected automation after confirmation", async () => {
    const automation = makeAutomation();
    mocks.automationsState.data = { automations: [automation], total: 1 };
    const user = userEvent.setup();

    renderList();
    await user.click(
      screen.getByRole("button", { name: `delete-${automation.id}` }),
    );
    expect(screen.getByTestId("delete-modal")).toHaveTextContent(
      automation.name,
    );
    await user.click(screen.getByRole("button", { name: "confirm-delete" }));

    expect(mocks.remove).toHaveBeenCalledWith(automation.id);
    expect(screen.queryByTestId("delete-modal")).not.toBeInTheDocument();
  });

  it("lets the user cancel deletion", async () => {
    const automation = makeAutomation();
    mocks.automationsState.data = { automations: [automation], total: 1 };
    const user = userEvent.setup();

    renderList();
    await user.click(
      screen.getByRole("button", { name: `delete-${automation.id}` }),
    );
    await user.click(screen.getByRole("button", { name: "cancel-delete" }));

    expect(mocks.remove).not.toHaveBeenCalled();
    expect(screen.queryByTestId("delete-modal")).not.toBeInTheDocument();
  });

  it("opens and closes editing for local automations", async () => {
    const automation = makeAutomation();
    mocks.automationsState.data = { automations: [automation], total: 1 };
    const user = userEvent.setup();

    renderList();
    await user.click(
      screen.getByRole("button", { name: `edit-${automation.id}` }),
    );
    expect(screen.getByTestId("edit-modal")).toHaveTextContent(automation.name);
    await user.click(screen.getByRole("button", { name: "close-edit" }));

    expect(screen.queryByTestId("edit-modal")).not.toBeInTheDocument();
  });

  it("offers editing regardless of the backend kind", () => {
    const automation = makeAutomation();
    mocks.backendKind = "cloud";
    mocks.automationsState.data = { automations: [automation], total: 1 };

    renderList();

    expect(
      screen.getByRole("button", { name: `edit-${automation.id}` }),
    ).toBeInTheDocument();
  });

  it("downloads an exported automation and records the backend kind", async () => {
    const automation = makeAutomation();
    mocks.automationsState.data = { automations: [automation], total: 1 };
    const user = userEvent.setup();

    renderList();
    await user.click(
      screen.getByRole("button", { name: `export-${automation.id}` }),
    );

    expect(mocks.serializeAutomation).toHaveBeenCalledWith(automation);
    expect(mocks.exportFilename).toHaveBeenCalledWith(automation);
    expect(mocks.downloadBlob).toHaveBeenCalledWith(
      expect.any(Blob),
      "daily-digest.automation.json",
    );
    const blob = mocks.downloadBlob.mock.calls[0][0] as Blob;
    expect(blob.type).toBe("application/json");
    expect(await readBlobText(blob)).toBe(
      `${JSON.stringify(mocks.serializeAutomation.mock.results[0].value, null, 2)}\n`,
    );
    expect(mocks.trackExported).toHaveBeenCalledWith({ backendKind: "local" });
  });

  it("opens and closes the add-automation form", async () => {
    const user = userEvent.setup();
    renderList();

    expect(screen.queryByTestId("add-modal")).not.toBeInTheDocument();
    await user.click(screen.getByTestId("automations-add-automation"));
    expect(screen.getByTestId("add-modal")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "close-add" }));

    expect(screen.queryByTestId("add-modal")).not.toBeInTheDocument();
  });

  it("opens the import modal from the import action", async () => {
    const user = userEvent.setup();
    renderList();

    expect(screen.queryByTestId("import-modal")).not.toBeInTheDocument();
    await user.click(screen.getByTestId("automations-import-automation"));

    expect(screen.getByTestId("import-modal")).toBeInTheDocument();
    expect(screen.getByTestId("automations-import-file")).toBeInTheDocument();
  });

  it("rejects malformed JSON before validating an import", async () => {
    renderList();
    uploadFile("not json");

    await waitFor(() => {
      expect(mocks.errorToast).toHaveBeenCalledWith(
        I18nKey.AUTOMATIONS$IMPORT_INVALID_JSON,
      );
    });
    expect(mocks.parseAutomationFile).not.toHaveBeenCalled();
  });

  it.each([
    { error: new Error("invalid automation"), expected: "invalid automation" },
    { error: "unknown validation failure", expected: I18nKey.ERROR$GENERIC },
  ])("surfaces import validation failures", async ({ error, expected }) => {
    mocks.parseAutomationFile.mockImplementation(() => {
      throw error;
    });
    renderList();

    uploadFile("{}");

    await waitFor(() => {
      expect(mocks.errorToast).toHaveBeenCalledWith(expected);
    });
  });

  it("previews a valid import and lets the user close it", async () => {
    const spec = makeImportSpec({ name: "Preview me" });
    mocks.parseAutomationFile.mockReturnValue(spec);
    const user = userEvent.setup();
    renderList();

    uploadFile("{}");
    expect(await screen.findByTestId("import-modal")).toHaveTextContent(
      "Preview me",
    );
    await user.click(screen.getByRole("button", { name: "close-import" }));

    expect(screen.queryByTestId("import-modal")).not.toBeInTheDocument();
  });

  it("imports a preview as disabled and links to the created automation", async () => {
    const spec = makeImportSpec({ name: "Preview me" });
    mocks.parseAutomationFile.mockReturnValue(spec);
    mocks.importState.isPending = true;
    mocks.importAutomation.mockImplementation(
      (
        _spec: AutomationSpec,
        options: { onSuccess: (created: Automation) => void },
      ) =>
        options.onSuccess(
          makeAutomation({ id: "created/id", name: "Created automation" }),
        ),
    );
    const user = userEvent.setup();
    renderList();

    uploadFile("{}");
    const modal = await screen.findByTestId("import-modal");
    expect(modal).toHaveAttribute("data-importing", "true");
    await user.click(screen.getByRole("button", { name: "confirm-import" }));

    expect(mocks.importAutomation).toHaveBeenCalledWith(
      { ...spec, enabled: false },
      expect.objectContaining({
        onSuccess: expect.any(Function),
        onError: expect.any(Function),
      }),
    );
    expect(mocks.successToastWithLink).toHaveBeenCalledWith(
      I18nKey.AUTOMATIONS$IMPORT_SUCCESS,
      I18nKey.AUTOMATIONS$IMPORT_VIEW,
      "/automations/created%2Fid",
    );
    expect(mocks.translate).toHaveBeenCalledWith(
      I18nKey.AUTOMATIONS$IMPORT_SUCCESS,
      { name: "Created automation" },
    );
    expect(screen.queryByTestId("import-modal")).not.toBeInTheDocument();
  });

  it.each([
    {
      label: "a backend detail",
      error: Object.assign(new Error("axios fallback"), {
        isAxiosError: true,
        response: { data: { detail: "duplicate automation" } },
      }),
      expected: "duplicate automation",
    },
    {
      label: "an Axios message when detail is structured",
      error: Object.assign(new Error("request failed"), {
        isAxiosError: true,
        response: { data: { detail: { code: "duplicate" } } },
      }),
      expected: "request failed",
    },
    {
      label: "an Axios message when no response arrived",
      error: Object.assign(new Error("network failed"), {
        isAxiosError: true,
      }),
      expected: "network failed",
    },
    {
      label: "a non-Axios error message",
      error: new Error("hidden implementation detail"),
      expected: "hidden implementation detail",
    },
  ])("surfaces $label when import fails", async ({ error, expected }) => {
    mocks.importAutomation.mockImplementation(
      (
        _spec: AutomationSpec,
        options: { onError: (importError: unknown) => void },
      ) => options.onError(error),
    );
    const user = userEvent.setup();
    renderList();

    uploadFile("{}");
    await user.click(
      await screen.findByRole("button", { name: "confirm-import" }),
    );

    expect(mocks.errorToast).toHaveBeenCalledWith(expected);
  });

  it("persists view changes", async () => {
    mocks.automationsState.data = {
      automations: [makeAutomation()],
      total: 1,
    };
    window.localStorage.setItem("openhands-automations-view", "list");
    const user = userEvent.setup();

    renderList();
    expect(screen.getByTestId("view-toggle")).toHaveTextContent("list");
    await user.click(screen.getByTestId("view-toggle"));

    expect(screen.getByTestId("view-toggle")).toHaveTextContent("grid");
    expect(window.localStorage.getItem("openhands-automations-view")).toBe(
      "grid",
    );
  });

  it("loads the next page size while more automations remain", async () => {
    mocks.automationsState.data = {
      automations: [makeAutomation()],
      total: 51,
    };
    const user = userEvent.setup();

    renderList();
    await user.click(
      screen.getByRole("button", { name: I18nKey.AUTOMATIONS$LOAD_MORE }),
    );

    expect(mocks.useAutomations).toHaveBeenLastCalledWith({
      limit: 100,
      offset: 0,
      enabled: true,
    });
  });

  it("does not offer another page when the total is already displayed", () => {
    mocks.automationsState.data = {
      automations: [makeAutomation()],
      total: 1,
    };

    renderList();

    expect(
      screen.queryByRole("button", { name: I18nKey.AUTOMATIONS$LOAD_MORE }),
    ).not.toBeInTheDocument();
  });
});
