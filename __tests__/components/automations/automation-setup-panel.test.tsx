import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HttpError } from "@openhands/typescript-client";
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
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import {
  setAutomationSetupDraft,
  type AutomationSetupDraft,
} from "#/api/automation-setup-draft-store";
import { packTarGzip } from "#/utils/tar-gzip";
import { handleAutomationFormUpdateAction } from "#/services/automation-form";
import { AUTOMATION_FORM_UPDATE_ACTION_KIND } from "#/constants/automation-form";
import { GitProviderItemsService } from "#/api/git-provider-items-service";
import { AUTOMATION_SETUP_SHOW_AGENT_EVENT } from "#/components/features/automations/setup/automation-setup-agent-request";
import { useDeploymentCapabilities } from "#/hooks/query/use-manifest-capabilities";
import type { AutomationDraftApiResponse } from "#/manifests/types";

const mockNavigate = vi.fn();
const mockToastSuccess = vi.fn();
const { mockSendMessage } = vi.hoisted(() => ({
  mockSendMessage: vi.fn().mockResolvedValue({ queued: false }),
}));

vi.mock("react-hot-toast", () => ({
  default: { success: (...args: unknown[]) => mockToastSuccess(...args) },
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
    i18n: { language: "en" },
  }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
}));

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: {
      updateConversationTags: vi.fn().mockResolvedValue({ tags: {} }),
    },
  }),
);

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
    listAutomationRuns: vi.fn().mockResolvedValue({ runs: [], total: 0 }),
    dispatchServerDraft: vi.fn(),
    createCustomWebhook: vi.fn(),
    supportsAutomationDrafts: vi.fn(),
  },
}));

vi.mock("#/hooks/use-send-message", () => ({
  useSendMessage: () => ({ send: mockSendMessage }),
}));

vi.mock("#/hooks/query/use-manifest-capabilities", () => ({
  useDeploymentCapabilities: vi.fn(() => ({ data: null, isLoading: false })),
}));

vi.mock("#/hooks/query/use-agent-profiles", () => ({
  useAgentProfiles: () => ({
    data: {
      profiles: [
        {
          id: "agent-default",
          name: "default",
          agent_kind: "openhands",
          revision: 1,
          llm_profile_ref: "claude-opus-4-5-20251101",
          mcp_server_refs: null,
        },
        {
          id: "agent-reviewer",
          name: "reviewer",
          agent_kind: "openhands",
          revision: 1,
          llm_profile_ref: "fast",
          mcp_server_refs: null,
        },
      ],
      active_agent_profile_id: "agent-default",
    },
    isLoading: false,
  }),
}));

vi.mock("#/api/git-provider-items-service", () => ({
  GitProviderItemsService: {
    listUserRepositories: vi.fn(async () => [
      "OpenHands/OpenHands",
      "OpenHands/software-agent-sdk",
    ]),
  },
}));

vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: () => ({
    providers: ["github"],
    isLoadingSettings: false,
  }),
}));

vi.mock("#/hooks/query/use-git-repositories", () => ({
  useGitRepositories: () => ({
    data: {
      pages: [
        {
          items: [
            {
              id: "1",
              full_name: "OpenHands/OpenHands",
              git_provider: "github",
              is_public: true,
            },
            {
              id: "2",
              full_name: "OpenHands/software-agent-sdk",
              git_provider: "github",
              is_public: true,
            },
          ],
          next_page_id: null,
        },
      ],
    },
    isLoading: false,
    isError: false,
    hasNextPage: false,
    isFetchingNextPage: false,
    fetchNextPage: vi.fn(),
    onLoadMore: vi.fn(),
  }),
}));

vi.mock("#/hooks/query/use-llm-profiles", () => ({
  useLlmProfiles: () => ({
    data: {
      profiles: [
        { name: "fast", model: "openai/gpt-4.1-mini" },
        {
          name: "claude-opus-4-5-20251101",
          model: "anthropic/claude-opus-4-5",
        },
      ],
      active_profile: "claude-opus-4-5-20251101",
    },
    isLoading: false,
  }),
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
  conversationTags: Record<string, string> | null = null,
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
        conversationTags={conversationTags}
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
    vi.mocked(
      AgentServerConversationService.updateConversationTags,
    ).mockResolvedValue({ tags: {} } as never);
    vi.mocked(GitProviderItemsService.listUserRepositories).mockResolvedValue([
      "OpenHands/OpenHands",
      "OpenHands/software-agent-sdk",
    ]);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("shows a one-time date when Once is selected and a time of day otherwise", async () => {
    const user = userEvent.setup();
    renderPanel();

    expect(
      screen.getByTestId("automation-setup-frequency-daily"),
    ).toHaveAttribute("aria-checked", "true");
    expect(screen.getByTestId("automation-setup-time")).toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-setup-datetime"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByTestId("automation-setup-frequency-once"));

    expect(screen.getByTestId("automation-setup-datetime")).toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-setup-time"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("automation-setup-timezone")).toHaveValue(
      "America/New_York",
    );
  });

  it("switches between prompt and custom, and reveals plugin fields on the prompt page", async () => {
    const user = userEvent.setup();
    renderPanel();

    expect(screen.getByTestId("automation-setup-prompt")).toHaveValue(
      "Review every pull request",
    );
    expect(
      screen.queryByTestId("automation-setup-kind-plugin"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-setup-plugin-source"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByTestId("automation-setup-add-plugin"));
    expect(
      screen.getByTestId("automation-setup-plugin-source"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("automation-setup-prompt")).toBeInTheDocument();

    await user.click(screen.getByTestId("automation-setup-kind-custom"));
    expect(
      screen.queryByTestId("automation-setup-plugin-source"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-setup-add-plugin"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByText("AUTOMATION_SETUP$CUSTOM_PYTHON"),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("AUTOMATION_SETUP$PYTHON_CODE"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-setup-model"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("automation-setup-custom-code-grip"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("automation-setup-custom-code-scroll"),
    ).not.toHaveClass("p-4");
    expect(screen.getByTestId("automation-setup-custom-code")).toHaveClass(
      "p-0",
    );
    expect(
      screen.queryByTestId("automation-setup-entrypoint"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("automation-setup-custom-details-drawer"),
    ).toHaveClass("border-x", "border-b", "rounded-b-[15px]");
    await user.click(
      screen.getByTestId("automation-setup-custom-details-toggle"),
    );
    expect(
      screen.getByTestId("automation-setup-custom-details-drawer"),
    ).toContainElement(screen.getByTestId("automation-setup-entrypoint"));
    expect(screen.getByTestId("automation-setup-entrypoint")).toHaveValue(
      "python3 main.py",
    );
    expect(
      screen
        .getByTestId("automation-setup-custom-code")
        .compareDocumentPosition(
          screen.getByTestId("automation-setup-entrypoint"),
        ) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      screen
        .getByTestId("automation-setup-entrypoint")
        .compareDocumentPosition(
          screen.getByTestId("automation-setup-setup-script"),
        ) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
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
    expect(
      screen.queryByTestId("automation-setup-repository"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("automation-setup-add-timeout"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("automation-setup-kind-prompt"));
    expect(screen.getByTestId("automation-setup-prompt")).toBeInTheDocument();
    expect(
      screen.getByTestId("automation-setup-repository"),
    ).not.toHaveTextContent("COMMON$OPTIONAL");
    expect(screen.getByTestId("automation-setup-prompt-drawer")).toHaveClass(
      "border-x",
      "border-b",
    );
    expect(
      screen.queryByTestId("automation-setup-entrypoint"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("automation-setup-plugin-source"),
    ).toBeInTheDocument();
  });

  it("removes an opened timeout or plugin from additional options", async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-add-timeout"));
    expect(screen.getByTestId("automation-setup-timeout")).toBeInTheDocument();
    await user.click(screen.getByTestId("automation-setup-timeout-remove"));
    expect(
      screen.queryByTestId("automation-setup-timeout"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("automation-setup-add-timeout"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("automation-setup-add-plugin"));
    expect(
      screen.getByTestId("automation-setup-plugin-source"),
    ).toBeInTheDocument();
    await user.click(screen.getByTestId("automation-setup-plugin-remove"));
    expect(
      screen.queryByTestId("automation-setup-plugin-source"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("automation-setup-add-plugin"),
    ).toBeInTheDocument();
  });

  it.each([
    {
      label: "an axios-shaped 404 (local backend)",
      error: { response: { status: 404 } },
    },
    {
      label: "the shared client's HttpError 404 (cloud backend)",
      error: new HttpError(404, "Not Found", { detail: "No such route" }),
    },
  ])(
    "falls back to validation when draft endpoints are unavailable ($label)",
    async ({ error }) => {
      vi.mocked(AutomationService.createServerDraft).mockRejectedValue(error);
      vi.mocked(AutomationService.validateDraft).mockResolvedValue({
        valid: true,
        errors: [],
      });

      const user = userEvent.setup();
      renderPanel();

      await user.click(screen.getByTestId("automation-setup-test"));

      await waitFor(() =>
        expect(AutomationService.createServerDraft).toHaveBeenCalledWith(
          expect.objectContaining({
            endpoint: "/v1/preset/prompt",
            draft: expect.objectContaining({
              enabled: false,
              prompt: "Review every pull request",
            }),
          }),
        ),
      );
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
      });
      expect(screen.getByTestId("automation-setup-status")).toHaveTextContent(
        "AUTOMATION_SETUP$READY_TO_TEST",
      );
    },
  );

  it("pins an automation model without changing the conversation profile", async () => {
    vi.mocked(AutomationService.createServerDraft).mockRejectedValue({
      response: { status: 404 },
    });
    vi.mocked(AutomationService.validateDraft).mockResolvedValue({
      valid: true,
      errors: [],
    });

    const user = userEvent.setup();
    renderPanel();

    expect(screen.getByTestId("automation-setup-model")).toHaveAttribute(
      "aria-label",
      "claude-opus-4-5-20251101",
    );
    await user.click(screen.getByTestId("automation-setup-model"));
    await user.click(screen.getByTestId("automation-setup-model-option-fast"));
    expect(screen.getByTestId("automation-setup-model")).toHaveAttribute(
      "aria-label",
      "fast",
    );

    await user.click(screen.getByTestId("automation-setup-test"));

    await waitFor(() =>
      expect(AutomationService.validateDraft).toHaveBeenCalledWith(
        expect.objectContaining({
          draft: expect.objectContaining({ model: "fast" }),
        }),
      ),
    );
  });

  it("pins an automation agent profile without changing the conversation", async () => {
    vi.mocked(AutomationService.createServerDraft).mockRejectedValue({
      response: { status: 404 },
    });
    vi.mocked(AutomationService.validateDraft).mockResolvedValue({
      valid: true,
      errors: [],
    });

    const user = userEvent.setup();
    renderPanel();

    expect(
      screen.getByTestId("automation-setup-agent-profile"),
    ).toHaveAttribute("aria-label", "default");
    await user.click(screen.getByTestId("automation-setup-agent-profile"));
    await user.click(
      screen.getByTestId("automation-setup-agent-profile-option-reviewer"),
    );
    expect(
      screen.getByTestId("automation-setup-agent-profile"),
    ).toHaveAttribute("aria-label", "reviewer");
    expect(screen.getByTestId("automation-setup-model")).toHaveAttribute(
      "aria-label",
      "claude-opus-4-5-20251101",
    );

    await user.click(screen.getByTestId("automation-setup-test"));

    await waitFor(() =>
      expect(AutomationService.validateDraft).toHaveBeenCalledWith(
        expect.objectContaining({
          draft: expect.objectContaining({
            agent_profile_id: "agent-reviewer",
          }),
        }),
      ),
    );
  });

  it("sends each comma-separated repository to the automation service", async () => {
    vi.mocked(AutomationService.createServerDraft).mockRejectedValue({
      response: { status: 404 },
    });
    vi.mocked(AutomationService.validateDraft).mockResolvedValue({
      valid: true,
      errors: [],
    });

    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-repository-add"));
    await user.click(screen.getByTestId("automation-setup-repository-custom"));
    await user.type(
      screen.getByTestId("automation-setup-repository-address"),
      "OpenHands/OpenHands",
    );
    await user.click(screen.getByTestId("automation-setup-repository-submit"));

    const addButton = screen.getByTestId("automation-setup-repository-add");
    const pill = screen.getByTestId("automation-setup-repository-value");
    expect(screen.getByTestId("automation-setup-repository-values")).toHaveClass(
      "flex-wrap",
    );
    expect(screen.getByTestId("automation-setup-kind-custom")).toHaveClass(
      "bg-tertiary",
      "text-content",
    );
    expect(
      addButton.compareDocumentPosition(pill) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();

    await user.click(screen.getByTestId("automation-setup-repository-add"));
    await user.click(screen.getByTestId("automation-setup-repository-custom"));
    await user.type(
      screen.getByTestId("automation-setup-repository-address"),
      "OpenHands/software-agent-sdk",
    );
    await user.click(screen.getByTestId("automation-setup-repository-submit"));
    await user.click(screen.getByTestId("automation-setup-test"));

    await waitFor(() =>
      expect(AutomationService.validateDraft).toHaveBeenCalledWith(
        expect.objectContaining({
          draft: expect.objectContaining({
            repos: [
              { url: "OpenHands/OpenHands", provider: "github" },
              { url: "OpenHands/software-agent-sdk", provider: "github" },
            ],
          }),
        }),
      ),
    );
  });

  it("adds a listed repository and keeps custom pinned to the address modal", async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-repository-add"));
    const menu = screen.getByTestId("automation-setup-repository-menu");
    const custom = screen.getByTestId("automation-setup-repository-custom");
    expect(menu.lastElementChild).toContainElement(custom);
    expect(menu.className).toContain("top-full");
    expect(
      screen.queryByTestId("automation-setup-add-repository-modal"),
    ).not.toBeInTheDocument();

    await user.click(
      await screen.findByTestId(
        "automation-setup-repository-option-OpenHands/OpenHands",
      ),
    );
    expect(
      screen.getByTestId("automation-setup-repository-value"),
    ).toHaveTextContent("OpenHands/OpenHands");

    await user.click(screen.getByTestId("automation-setup-repository-add"));
    await user.click(screen.getByTestId("automation-setup-repository-custom"));
    expect(
      screen.getByTestId("automation-setup-add-repository-modal"),
    ).toBeInTheDocument();
  });

  it("filters the repository menu from the search and keeps a divider above custom", async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-repository-add"));
    expect(
      await screen.findByTestId(
        "automation-setup-repository-option-OpenHands/OpenHands",
      ),
    ).toBeInTheDocument();
    const custom = screen.getByTestId("automation-setup-repository-custom");
    const divider = screen.getByTestId(
      "automation-setup-repository-custom-divider",
    );
    expect(
      divider.compareDocumentPosition(custom) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();

    await user.type(
      screen.getByTestId("automation-setup-repository-search"),
      "sdk",
    );

    expect(
      screen.queryByTestId(
        "automation-setup-repository-option-OpenHands/OpenHands",
      ),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId(
        "automation-setup-repository-option-OpenHands/software-agent-sdk",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("automation-setup-repository-custom"),
    ).toBeInTheDocument();
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
    ).toHaveAttribute("aria-checked", "true");
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

  it("shows a weekday dropdown for a weekly schedule", async () => {
    vi.mocked(AutomationService.createServerDraft).mockRejectedValue({
      response: { status: 404 },
    });
    vi.mocked(AutomationService.validateDraft).mockResolvedValue({
      valid: true,
      errors: [],
    });

    const user = userEvent.setup();
    renderPanel();

    expect(
      screen.queryByTestId("automation-setup-weekday"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByTestId("automation-setup-frequency-weekly"));
    const weekday = screen.getByTestId("automation-setup-weekday");
    expect(weekday).toHaveValue("1");
    expect(weekday.parentElement?.parentElement).toHaveTextContent(
      "AUTOMATION_SETUP$ON",
    );
    await user.selectOptions(weekday, "3");

    await user.click(screen.getByTestId("automation-setup-test"));

    await waitFor(() =>
      expect(AutomationService.validateDraft).toHaveBeenCalledWith(
        expect.objectContaining({
          draft: expect.objectContaining({
            trigger: expect.objectContaining({ schedule: "0 9 * * 3" }),
          }),
        }),
      ),
    );
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

  it("hides the event filter until Add Event Filter is clicked", async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByText("AUTOMATIONS$DETAIL$TRIGGER_EVENT"));
    expect(
      screen.queryByTestId("automation-setup-event-filter"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("automation-setup-event-key").parentElement,
    ).toContainElement(screen.getByTestId("automation-setup-add-event-filter"));
    expect(screen.getByTestId("automation-setup-add-event-filter")).toHaveTextContent(
      "AUTOMATION_SETUP$ADD_FILTER",
    );

    await user.click(screen.getByTestId("automation-setup-add-event-filter"));
    expect(
      screen.getByTestId("automation-setup-event-filter"),
    ).toBeInTheDocument();

    await user.click(
      screen.getByTestId("automation-setup-event-filter-remove"),
    );
    expect(
      screen.queryByTestId("automation-setup-event-filter"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("automation-setup-add-event-filter"),
    ).toBeInTheDocument();
  });

  it("asks the agent to fill the event trigger", async () => {
    const user = userEvent.setup();
    const showAgent = vi.fn();
    window.addEventListener(AUTOMATION_SETUP_SHOW_AGENT_EVENT, showAgent);
    renderPanel();

    await user.click(screen.getByText("AUTOMATIONS$DETAIL$TRIGGER_EVENT"));
    await user.click(screen.getByTestId("automation-setup-ask-agent"));

    expect(mockSendMessage).toHaveBeenCalledWith({
      action: "message",
      args: { content: "AUTOMATION_SETUP$ASK_AGENT_EVENT_PROMPT" },
    });
    expect(showAgent).toHaveBeenCalled();
    window.removeEventListener(AUTOMATION_SETUP_SHOW_AGENT_EVENT, showAgent);
  });

  it("shows a spinner in the repository menu while the list loads", async () => {
    let resolveNames: (names: string[]) => void = () => {};
    vi.mocked(GitProviderItemsService.listUserRepositories).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveNames = resolve;
        }),
    );
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-repository-add"));
    expect(
      screen.getByTestId("automation-setup-repository-loading"),
    ).toBeInTheDocument();

    resolveNames(["OpenHands/OpenHands"]);
    expect(
      await screen.findByTestId(
        "automation-setup-repository-option-OpenHands/OpenHands",
      ),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-setup-repository-loading"),
    ).not.toBeInTheDocument();
  });

  it("removes one plugin row from the module", async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-add-plugin"));
    await user.click(screen.getByTestId("automation-setup-add-plugin"));
    expect(
      screen.getByTestId("automation-setup-plugin-source-1"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("automation-setup-plugin-remove-1"));

    expect(
      screen.queryByTestId("automation-setup-plugin-source-1"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("automation-setup-plugin-source"),
    ).toBeInTheDocument();
  });

  it("creates a draft with every plugin that has a source", async () => {
    vi.mocked(AutomationService.createAutomationDraft).mockResolvedValue({
      id: "automation-plugins",
    });

    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-add-plugin"));
    expect(screen.getByTestId("automation-setup-plugin-ref")).toHaveValue(
      "main",
    );
    expect(screen.getByTestId("automation-setup-plugin-remove")).toHaveClass(
      "size-6",
    );
    await user.type(
      screen.getByTestId("automation-setup-plugin-source"),
      "github:org/blockers-plugin",
    );
    await user.click(screen.getByTestId("automation-setup-add-plugin"));
    const pluginModule = screen.getByTestId("automation-setup-plugin-module");
    expect(pluginModule).toContainElement(
      screen.getByTestId("automation-setup-add-plugin"),
    );
    expect(pluginModule).toContainElement(
      screen.getByTestId("automation-setup-plugin-source-1"),
    );
    expect(
      within(pluginModule).getAllByText("AUTOMATION_SETUP$PLUGIN_SOURCE"),
    ).toHaveLength(1);
    expect(
      within(pluginModule).getAllByText("AUTOMATION_SETUP$PLUGIN_REF"),
    ).toHaveLength(1);
    expect(
      screen.getByTestId("automation-setup-plugin-ref").compareDocumentPosition(
        screen.getByTestId("automation-setup-plugin-remove"),
      ) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(screen.getByTestId("automation-setup-plugin-ref-1")).toHaveValue(
      "main",
    );
    await user.type(
      screen.getByTestId("automation-setup-plugin-source-1"),
      "github:org/review-plugin",
    );

    await user.click(screen.getByTestId("automation-setup-create"));

    await waitFor(() =>
      expect(AutomationService.createAutomationDraft).toHaveBeenCalledWith(
        expect.objectContaining({
          plugins: [
            { source: "github:org/blockers-plugin", ref: "main" },
            { source: "github:org/review-plugin", ref: "main" },
          ],
        }),
        "plugin",
      ),
    );
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

    await user.click(
      screen.getByTestId("automation-setup-custom-details-toggle"),
    );
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

    it("tags the conversation as automation setup mode when opened", async () => {
      renderPanel();

      await waitFor(() =>
        expect(
          AgentServerConversationService.updateConversationTags,
        ).toHaveBeenCalledWith(
          "conv-1",
          expect.objectContaining({ automationsetup: "draft" }),
        ),
      );
    });

    it("shows the save state beside Save draft instead of in its own bar", () => {
      renderPanel();

      const label = screen.getByTestId("automation-setup-save-state");
      const save = screen.getByTestId("automation-setup-save-draft");

      expect(label).toHaveTextContent("AUTOMATION_SETUP$UNSAVED_CHANGES");
      expect(label.className).not.toContain("border-b");
      expect(save.parentElement?.firstElementChild).toBe(label);
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
      expect(
        screen.getByTestId("automation-setup-draft-details"),
      ).toBeInTheDocument();
      expect(
        screen.getByTestId("automation-setup-draft-validity"),
      ).toHaveTextContent("AUTOMATION_SETUP$READY_TO_TEST");

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

    it("tags the conversation with the server draft id after saving", async () => {
      vi.mocked(AutomationService.createServerDraft).mockResolvedValue(
        dispatchableDraft,
      );

      const user = userEvent.setup();
      renderPanel(undefined, "conv-1", { existing: "tag" });

      await user.click(screen.getByTestId("automation-setup-save-draft"));

      await waitFor(() =>
        expect(
          AgentServerConversationService.updateConversationTags,
        ).toHaveBeenCalledWith(
          "conv-1",
          expect.objectContaining({
            existing: "tag",
            automationsetup: "draft",
            automationdraftid: "draft-1",
          }),
        ),
      );
    });

    it("hydrates a tagged server draft when the panel opens", async () => {
      vi.mocked(AutomationService.getServerDraft).mockResolvedValue({
        ...dispatchableDraft,
        name: "Saved Tagged Draft",
        draft: {
          prompt: "Use the persisted draft body",
          trigger: {
            type: "cron",
            schedule: "0 12 * * *",
            timezone: "UTC",
          },
        },
      });

      renderPanel({ prompt: "", kind: "prompt" }, "conv-1", {
        automationdraftid: "draft-1",
      });

      await waitFor(() =>
        expect(AutomationService.getServerDraft).toHaveBeenCalledWith(
          "draft-1",
        ),
      );
      expect(screen.getByTestId("automation-setup-name")).toHaveValue(
        "Saved Tagged Draft",
      );
      expect(screen.getByTestId("automation-setup-prompt")).toHaveValue(
        "Use the persisted draft body",
      );
      expect(
        screen.getByTestId("automation-setup-draft-details"),
      ).toBeInTheDocument();
    });

    it("shows a missing-draft message and creates a fresh draft after a tagged draft was deleted", async () => {
      vi.mocked(AutomationService.getServerDraft).mockRejectedValue({
        response: { status: 404 },
      });
      vi.mocked(AutomationService.createServerDraft).mockResolvedValue({
        ...dispatchableDraft,
        id: "draft-2",
      });

      const user = userEvent.setup();
      renderPanel(
        { prompt: "Recreate the automation", kind: "prompt" },
        "conv-1",
        { automationdraftid: "draft-missing" },
      );

      await screen.findByTestId("automation-setup-draft-missing");
      await user.click(screen.getByTestId("automation-setup-save-draft"));

      await waitFor(() =>
        expect(AutomationService.createServerDraft).toHaveBeenCalled(),
      );
      expect(AutomationService.updateServerDraft).not.toHaveBeenCalledWith(
        "draft-missing",
        expect.anything(),
      );
      expect(
        AgentServerConversationService.updateConversationTags,
      ).toHaveBeenCalledWith(
        "conv-1",
        expect.objectContaining({ automationdraftid: "draft-2" }),
      );
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
      expect(mockNavigate).not.toHaveBeenCalledWith(
        "/conversations/conv-run-1",
      );
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

    it("shows a public URL notice and collapses the test payload for event triggers", async () => {
      const user = userEvent.setup();
      renderPanel();

      expect(
        screen.queryByTestId("automation-setup-event-public-url-notice"),
      ).not.toBeInTheDocument();

      await user.click(screen.getByText("AUTOMATIONS$DETAIL$TRIGGER_EVENT"));

      expect(
        screen.getByTestId("automation-setup-event-public-url-notice"),
      ).toHaveTextContent("AUTOMATION_SETUP$EVENT_PUBLIC_URL_NOTICE");
      expect(
        screen.getByTestId("automation-setup-event-public-url-notice"),
      ).not.toHaveClass("bg-tertiary");
      expect(
        screen
          .getByTestId("automation-setup-event-source")
          .compareDocumentPosition(
            screen.getByTestId("automation-setup-event-public-url-notice"),
          ) & Node.DOCUMENT_POSITION_FOLLOWING,
      ).toBeTruthy();
      expect(
        screen.queryByTestId("automation-setup-event-test-payload"),
      ).not.toBeInTheDocument();

      await user.click(
        screen.getByTestId("automation-setup-event-test-payload-toggle"),
      );
      expect(
        screen.getByTestId("automation-setup-event-test-payload"),
      ).toBeInTheDocument();

      await user.click(
        screen.getByTestId("automation-setup-event-test-payload-toggle"),
      );
      expect(
        screen.queryByTestId("automation-setup-event-test-payload"),
      ).not.toBeInTheDocument();
    });

    it("sends synthetic JSON payload when testing an event draft", async () => {
      vi.mocked(AutomationService.createServerDraft).mockResolvedValue(
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
        automation_id: "auto-draft-1",
      } as never);

      const user = userEvent.setup();
      renderPanel();

      await user.click(screen.getByText("AUTOMATIONS$DETAIL$TRIGGER_EVENT"));
      await user.click(
        screen.getByTestId("automation-setup-event-test-payload-toggle"),
      );
      const payloadInput = await screen.findByTestId(
        "automation-setup-event-test-payload",
      );
      fireEvent.change(payloadInput, {
        target: {
          value: JSON.stringify({ type: "issue.created", action: "opened" }),
        },
      });
      await user.click(screen.getByTestId("automation-setup-test"));

      await waitFor(() =>
        expect(AutomationService.dispatchServerDraft).toHaveBeenCalledWith(
          "draft-1",
          { eventPayload: { type: "issue.created", action: "opened" } },
        ),
      );
    });

    it("registers a custom webhook source before testing an event draft", async () => {
      vi.mocked(useDeploymentCapabilities).mockReturnValue({
        data: {
          ready: true,
          features: ["automationDrafts", "webhookDelivery"],
          triggerKinds: ["cron", "event"],
          eventSources: ["github", "linear"],
          eventTypes: ["issues.*", "issue_comment.created"],
          triggers: {},
        },
        isLoading: false,
      } as never);
      vi.mocked(AutomationService.createCustomWebhook).mockResolvedValue({
        id: "webhook-1",
        org_id: "org-1",
        name: "Incident webhook",
        source: "incident-alerts",
        webhook_url:
          "https://app.all-hands.dev/v1/events/org-1/incident-alerts",
        event_key_expr: "event.type",
        signature_header: "X-Incident-Signature",
        signature_scheme: "hmac_sha256_hex",
        enabled: true,
        created_at: "2026-01-01T00:00:00.000Z",
        updated_at: "2026-01-01T00:00:00.000Z",
        webhook_secret: "generated-secret",
      });
      vi.mocked(AutomationService.createServerDraft).mockResolvedValue(
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
        automation_id: "auto-draft-1",
      } as never);

      const user = userEvent.setup();
      renderPanel();

      await user.click(screen.getByText("AUTOMATIONS$DETAIL$TRIGGER_EVENT"));
      expect(
        screen.queryByTestId("automation-setup-custom-webhook-enabled"),
      ).not.toBeInTheDocument();
      expect(screen.getByTestId("automation-setup-event-source")).toHaveValue(
        "GitHub",
      );
      await user.click(screen.getByTestId("automation-setup-event-source"));
      expect(
        screen.queryByTestId("automation-setup-event-source-menu"),
      ).not.toBeInTheDocument();
      await user.click(
        screen.getByTestId("automation-setup-event-source-toggle"),
      );
      expect(
        screen.getByTestId("automation-setup-event-source-option-linear"),
      ).toBeInTheDocument();
      const custom = screen.getByTestId("automation-setup-event-source-custom");
      expect(
        screen
          .getByTestId("automation-setup-event-source-option-github")
          .compareDocumentPosition(custom) & Node.DOCUMENT_POSITION_FOLLOWING,
      ).toBeTruthy();
      expect(
        document.querySelector(
          '#automation-setup-event-key-options option[value="issues.*"]',
        ),
      ).not.toBeNull();
      await user.click(custom);
      fireEvent.change(screen.getByTestId("automation-setup-event-source"), {
        target: { value: "incident-alerts" },
      });
      expect(
        screen.getByTestId("automation-setup-custom-webhook-enabled"),
      ).toBeInTheDocument();
      await user.click(screen.getByTestId("automation-setup-event-source-toggle"));
      expect(
        screen.getByTestId("automation-setup-event-source-option-github"),
      ).toBeInTheDocument();
      await user.click(screen.getByTestId("automation-setup-event-source-toggle"));
      await user.click(
        screen.getByTestId("automation-setup-custom-webhook-enabled"),
      );
      await user.type(
        screen.getByTestId("automation-setup-custom-webhook-name"),
        "Incident webhook",
      );
      fireEvent.change(
        screen.getByTestId("automation-setup-custom-webhook-event-key-expr"),
        { target: { value: "event.type" } },
      );
      fireEvent.change(
        screen.getByTestId("automation-setup-custom-webhook-signature-header"),
        { target: { value: "X-Incident-Signature" } },
      );
      await user.click(screen.getByTestId("automation-setup-test"));

      await waitFor(() =>
        expect(AutomationService.createCustomWebhook).toHaveBeenCalledWith({
          name: "Incident webhook",
          source: "incident-alerts",
          event_key_expr: "event.type",
          signature_header: "X-Incident-Signature",
          signature_scheme: "hmac_sha256_hex",
        }),
      );
      expect(AutomationService.createServerDraft).toHaveBeenCalledWith(
        expect.objectContaining({
          draft: expect.objectContaining({
            trigger: expect.objectContaining({
              type: "event",
              source: "incident-alerts",
            }),
          }),
        }),
      );
    });

    it("requires valid JSON before testing an event draft", async () => {
      const user = userEvent.setup();
      renderPanel();

      await user.click(screen.getByText("AUTOMATIONS$DETAIL$TRIGGER_EVENT"));
      await user.click(
        screen.getByTestId("automation-setup-event-test-payload-toggle"),
      );
      const payloadInput = await screen.findByTestId(
        "automation-setup-event-test-payload",
      );
      fireEvent.change(payloadInput, { target: { value: "not json" } });
      await user.click(screen.getByTestId("automation-setup-test"));

      expect(screen.getByTestId("automation-setup-status")).toHaveTextContent(
        "AUTOMATION_SETUP$TEST_EVENT_PAYLOAD_INVALID",
      );
      expect(AutomationService.createServerDraft).not.toHaveBeenCalled();
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
      expect(
        AgentServerConversationService.updateConversationTags,
      ).toHaveBeenCalledWith(
        "conv-1",
        expect.not.objectContaining({ automationdraftid: expect.any(String) }),
      );
      expect(mockNavigate).toHaveBeenCalledWith(
        "/automations/automation-final",
      );
    });
  });
});
