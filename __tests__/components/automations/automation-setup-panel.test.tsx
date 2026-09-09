import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import {
  AGENT_FIELD_STREAM_CHARACTER_DELAY_MS,
  AGENT_FIELD_STREAM_SETTLE_DELAY_MS,
  AutomationSetupPanel,
} from "#/components/features/automations/setup/automation-setup-panel";
import AutomationService from "#/api/automation-service/automation-service.api";
import {
  setAutomationSetupDraft,
  type AutomationSetupDraft,
} from "#/api/automation-setup-draft-store";
import { packTarGzip } from "#/utils/tar-gzip";
import { handleAutomationFormUpdateAction } from "#/services/automation-form";
import { AUTOMATION_FORM_UPDATE_ACTION_KIND } from "#/constants/automation-form";
import { useDeploymentCapabilities } from "#/hooks/query/use-manifest-capabilities";
import type { AutomationDraftApiResponse } from "#/manifests/types";

const mockNavigate = vi.fn();
const mockToastSuccess = vi.fn();

vi.mock("react-hot-toast", () => ({
  default: { success: (...args: unknown[]) => mockToastSuccess(...args) },
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
}));

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    validateDraft: vi.fn(),
    createAutomationDraft: vi.fn(),
    uploadAutomationTarball: vi.fn(),
    createServerDraft: vi.fn(),
    updateServerDraft: vi.fn(),
    getServerDraft: vi.fn(),
    deleteServerDraft: vi.fn(),
    listServerDrafts: vi.fn(),
    dispatchServerDraft: vi.fn(),
    supportsAutomationDrafts: vi.fn(),
  },
}));

vi.mock("#/hooks/query/use-manifest-capabilities", () => ({
  useDeploymentCapabilities: vi.fn(() => ({ data: null, isLoading: false })),
}));

vi.mock("#/utils/tar-gzip", () => ({
  packTarGzip: vi.fn().mockResolvedValue(new Uint8Array([1, 2, 3])),
}));

vi.mock("#/manifests/automation-interface", () => ({
  automationDetailPath: (id: string) => `/automations/${id}`,
  getAutomationEndpoint: (name: string) =>
    name === "createPlugin"
      ? "/v1/preset/plugin"
      : name === "createBundle"
        ? "/v1"
        : "/v1/preset/prompt",
}));

function renderPanel(
  draft: AutomationSetupDraft = {
    prompt: "Review every pull request",
    kind: "prompt",
  },
  conversationId = "conv-1",
) {
  const value: NavigationContextValue = {
    currentPath: `/conversations/${conversationId}`,
    conversationId,
    isNavigating: false,
    navigate: mockNavigate,
  };

  return render(
    <NavigationProvider value={value}>
      <AutomationSetupPanel
        draft={draft}
        conversationId={conversationId}
        onClose={vi.fn()}
      />
    </NavigationProvider>,
  );
}

describe("AutomationSetupPanel", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    vi.clearAllMocks();
    // By default the deployment does not advertise server-backed drafts, so
    // the panel falls back to the preflight-only Test path the existing
    // assertions cover. BE-draft tests override this per-test.
    vi.mocked(useDeploymentCapabilities).mockReturnValue({
      data: null,
      isLoading: false,
    } as never);
    vi.mocked(AutomationService.supportsAutomationDrafts).mockReturnValue(
      false,
    );
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("switches between prompt, plugin, and custom form types", async () => {
    const user = userEvent.setup();
    renderPanel();

    expect(screen.getByTestId("automation-setup-prompt")).toHaveValue(
      "Review every pull request",
    );

    await user.click(screen.getByTestId("automation-setup-kind-plugin"));
    expect(
      screen.getByTestId("automation-setup-plugin-source"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("automation-setup-kind-custom"));
    expect(screen.getByTestId("automation-setup-entrypoint")).toHaveValue(
      "python3 main.py",
    );
    expect(
      screen.getByTestId("automation-setup-setup-script-path"),
    ).toHaveValue("setup.sh");
    expect(
      (
        screen.getByTestId(
          "automation-setup-custom-code",
        ) as HTMLTextAreaElement
      ).value,
    ).toContain("Review every pull request");
    expect(screen.queryByText("AUTOMATIONS$TIMEZONE")).not.toBeInTheDocument();
    expect(screen.getByLabelText("AUTOMATIONS$TIMEZONE")).toBe(
      screen.getByTestId("automation-setup-timezone"),
    );
    expect(
      screen.queryByTestId("automation-setup-rendered-code"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-setup-prompt"),
    ).not.toBeInTheDocument();
  });

  it("validates the prompt draft against the automation service", async () => {
    vi.mocked(AutomationService.validateDraft).mockResolvedValue({
      valid: true,
      errors: [],
    });

    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-test"));

    await waitFor(() =>
      expect(AutomationService.validateDraft).toHaveBeenCalledWith({
        endpoint: "/v1/preset/prompt",
        draft: expect.objectContaining({
          enabled: false,
          prompt: "Review every pull request",
          trigger: {
            type: "cron",
            schedule: "0 9 * * *",
            timezone: "America/New_York",
          },
        }),
      }),
    );
    expect(screen.getByTestId("automation-setup-status")).toHaveTextContent(
      "AUTOMATION_SETUP$TEST_PASSED",
    );
  });

  it("streams agent form updates field by field without overwriting user edits", async () => {
    vi.useFakeTimers();
    const conversationId = "conv-agent-updates";
    const draft: AutomationSetupDraft = {
      prompt: "Review every pull request",
      kind: "prompt",
    };
    setAutomationSetupDraft(conversationId, draft);
    renderPanel(draft, conversationId);

    await act(async () => {
      handleAutomationFormUpdateAction(
        {
          kind: AUTOMATION_FORM_UPDATE_ACTION_KIND,
          fields: {
            name: "PR Review Assistant",
            prompt: "Watch pull requests and draft review notes",
            frequency: "weekly",
            time: "10:30",
            timezone: "UTC",
          },
        },
        conversationId,
        "agent-event-1",
        "2026-01-01T00:00:00.000Z",
      );
    });

    const nameInput = screen.getByTestId("automation-setup-name");
    const promptInput = screen.getByTestId("automation-setup-prompt");
    expect(nameInput.closest("label")).toHaveAttribute(
      "data-streaming-active",
      "true",
    );
    expect(promptInput).toHaveValue("Review every pull request");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(AGENT_FIELD_STREAM_CHARACTER_DELAY_MS);
    });
    expect(nameInput).toHaveValue("P");
    expect(promptInput).toHaveValue("Review every pull request");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(
        "PR Review Assistant".length * AGENT_FIELD_STREAM_CHARACTER_DELAY_MS +
          AGENT_FIELD_STREAM_SETTLE_DELAY_MS +
          AGENT_FIELD_STREAM_CHARACTER_DELAY_MS,
      );
    });
    expect(nameInput).toHaveValue("PR Review Assistant");
    expect(promptInput.closest("label")).toHaveAttribute(
      "data-streaming-active",
      "true",
    );
    expect((promptInput as HTMLTextAreaElement).value).toMatch(/^W/);
    expect((promptInput as HTMLTextAreaElement).value).not.toBe(
      "Watch pull requests and draft review notes",
    );

    await act(async () => {
      await vi.runAllTimersAsync();
    });
    expect(promptInput).toHaveValue(
      "Watch pull requests and draft review notes",
    );
    expect(
      screen.getByTestId("automation-setup-frequency-weekly"),
    ).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByTestId("automation-setup-time")).toHaveValue("10:30");
    expect(screen.getByTestId("automation-setup-timezone")).toHaveValue("UTC");
    expect(screen.getByTestId("automation-setup-at-row")).toHaveTextContent(
      "AUTOMATION_SETUP$FILLED_BY_OPENHANDS",
    );
    expect(
      screen.getAllByText("AUTOMATION_SETUP$FILLED_BY_OPENHANDS").length,
    ).toBeGreaterThan(0);

    vi.useRealTimers();
    const realUser = userEvent.setup();
    await realUser.clear(nameInput);
    await realUser.type(nameInput, "Manual name");

    handleAutomationFormUpdateAction(
      {
        kind: AUTOMATION_FORM_UPDATE_ACTION_KIND,
        fields: { name: "Agent replacement" },
      },
      conversationId,
      "agent-event-2",
      "2026-01-01T00:00:01.000Z",
    );

    expect(nameInput).toHaveValue("Manual name");
  });

  it("creates plugin drafts with the selected plugin source", async () => {
    vi.mocked(AutomationService.createAutomationDraft).mockResolvedValue({
      id: "automation-1",
    });

    const user = userEvent.setup();
    renderPanel({
      prompt: "Summarize Slack blockers",
      kind: "plugin",
      plugins: ["github:org/blockers-plugin"],
    });

    await user.click(screen.getByTestId("automation-setup-create"));

    await waitFor(() =>
      expect(AutomationService.createAutomationDraft).toHaveBeenCalledWith(
        expect.objectContaining({
          enabled: false,
          prompt: "Summarize Slack blockers",
          plugins: [{ source: "github:org/blockers-plugin" }],
        }),
        "plugin",
      ),
    );
    expect(mockNavigate).toHaveBeenCalledWith("/automations/automation-1");
  });

  it("creates custom bundle drafts with entrypoint and setup script path", async () => {
    vi.mocked(AutomationService.uploadAutomationTarball).mockResolvedValue(
      "oh-internal://uploads/custom-archive",
    );
    vi.mocked(AutomationService.createAutomationDraft).mockResolvedValue({
      id: "automation-custom",
    });

    const user = userEvent.setup();
    renderPanel({
      prompt: "Run a custom security check",
      kind: "custom",
    });

    await user.clear(screen.getByTestId("automation-setup-entrypoint"));
    await user.type(
      screen.getByTestId("automation-setup-entrypoint"),
      "python3 main.py --once",
    );
    await user.click(screen.getByTestId("automation-setup-create"));

    await waitFor(() =>
      expect(AutomationService.uploadAutomationTarball).toHaveBeenCalledWith(
        "Run A Custom Security",
        new Uint8Array([1, 2, 3]),
      ),
    );
    expect(packTarGzip).toHaveBeenCalledWith([
      expect.objectContaining({ name: "main.py", mode: 0o644 }),
      expect.objectContaining({ name: "setup.sh", mode: 0o755 }),
    ]);
    expect(AutomationService.createAutomationDraft).toHaveBeenCalledWith(
      expect.objectContaining({
        tarball_path: "oh-internal://uploads/custom-archive",
        entrypoint: "python3 main.py --once",
        setup_script_path: "setup.sh",
      }),
      "custom",
    );
    expect(mockNavigate).toHaveBeenCalledWith("/automations/automation-custom");
  });

  describe("server-backed drafts (PR OpenHands/automation#417)", () => {
    const dispatchableDraft: AutomationDraftApiResponse = {
      id: "draft-1",
      endpoint: "/v1/preset/prompt",
      name: "Review Every Pull Request",
      draft: {} as never,
      validationErrors: null,
      dispatchable: true,
      sourceAutomationId: null,
      materializedAutomationId: null,
      lastTestRunId: null,
      createdAt: "2026-01-01T00:00:00.000Z",
      updatedAt: "2026-01-01T00:00:00.000Z",
    };

    beforeEach(() => {
      vi.mocked(AutomationService.supportsAutomationDrafts).mockReturnValue(
        true,
      );
      vi.mocked(useDeploymentCapabilities).mockReturnValue({
        data: {
          ready: true,
          features: ["automationDrafts"],
          triggerKinds: ["cron", "event"],
          eventSources: [],
          eventTypes: [],
          triggers: {},
        },
        isLoading: false,
      } as never);
    });

    it("creates a server draft on Save draft and updates it on the next save", async () => {
      vi.mocked(AutomationService.createServerDraft).mockResolvedValue({
        ...dispatchableDraft,
        validationErrors: null,
      });
      vi.mocked(AutomationService.updateServerDraft).mockResolvedValue({
        ...dispatchableDraft,
        updatedAt: "2026-01-01T00:00:01.000Z",
      });

      const user = userEvent.setup();
      renderPanel();

      await user.click(screen.getByTestId("automation-setup-save-draft"));

      await waitFor(() =>
        expect(AutomationService.createServerDraft).toHaveBeenCalledWith(
          expect.objectContaining({
            endpoint: "/v1/preset/prompt",
            draft: expect.objectContaining({
              prompt: "Review every pull request",
            }),
          }),
        ),
      );
      expect(screen.getByTestId("automation-setup-status")).toHaveTextContent(
        "AUTOMATION_SETUP$DRAFT_SAVED",
      );

      // Second save reuses the persisted id rather than creating again.
      await user.click(screen.getByTestId("automation-setup-save-draft"));
      await waitFor(() =>
        expect(AutomationService.updateServerDraft).toHaveBeenCalledWith(
          "draft-1",
          expect.objectContaining({ endpoint: "/v1/preset/prompt" }),
        ),
      );
      expect(AutomationService.createServerDraft).toHaveBeenCalledTimes(1);
    });

    it("persists then dispatches the draft on Test", async () => {
      vi.mocked(AutomationService.createServerDraft).mockResolvedValue(
        dispatchableDraft,
      );
      vi.mocked(AutomationService.updateServerDraft).mockResolvedValue(
        dispatchableDraft,
      );
      vi.mocked(AutomationService.dispatchServerDraft).mockResolvedValue({
        id: "run-1",
        status: "PENDING" as never,
        conversation_id: "conv-run-1",
        bash_command_id: null,
        error_detail: null,
        started_at: "2026-01-01T00:00:00.000Z",
        completed_at: null,
      });

      const user = userEvent.setup();
      renderPanel();

      await user.click(screen.getByTestId("automation-setup-test"));

      await waitFor(() =>
        expect(AutomationService.dispatchServerDraft).toHaveBeenCalledWith(
          "draft-1",
        ),
      );
      expect(AutomationService.validateDraft).not.toHaveBeenCalled();
      expect(screen.getByTestId("automation-setup-status")).toHaveTextContent(
        "AUTOMATION_SETUP$TEST_DISPATCHED",
      );
      expect(mockNavigate).toHaveBeenCalledWith("/conversations/conv-run-1");
    });

    it("surfaces validation errors when the draft is not dispatchable", async () => {
      vi.mocked(AutomationService.createServerDraft).mockResolvedValue({
        ...dispatchableDraft,
        dispatchable: false,
        validationErrors: [
          {
            field: "trigger.schedule",
            code: "interval_too_short",
            message: "Minimum interval is 5 minutes.",
          },
        ],
      });

      const user = userEvent.setup();
      renderPanel();

      await user.click(screen.getByTestId("automation-setup-test"));

      await waitFor(() =>
        expect(screen.getByTestId("automation-setup-status")).toHaveTextContent(
          "Minimum interval is 5 minutes.",
        ),
      );
      expect(AutomationService.dispatchServerDraft).not.toHaveBeenCalled();
    });

    it("deletes the persisted draft after a successful create", async () => {
      vi.mocked(AutomationService.createServerDraft).mockResolvedValue(
        dispatchableDraft,
      );
      vi.mocked(AutomationService.deleteServerDraft).mockResolvedValue();
      vi.mocked(AutomationService.createAutomationDraft).mockResolvedValue({
        id: "automation-final",
      });

      const user = userEvent.setup();
      renderPanel();

      // Save first so a draft id exists, then finalize.
      await user.click(screen.getByTestId("automation-setup-save-draft"));
      await waitFor(() =>
        expect(AutomationService.createServerDraft).toHaveBeenCalled(),
      );

      await user.click(screen.getByTestId("automation-setup-create"));

      await waitFor(() =>
        expect(AutomationService.deleteServerDraft).toHaveBeenCalledWith(
          "draft-1",
        ),
      );
      expect(mockNavigate).toHaveBeenCalledWith(
        "/automations/automation-final",
      );
    });
  });
});
