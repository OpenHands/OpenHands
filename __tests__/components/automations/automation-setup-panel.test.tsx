import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { compile } from "@tailwindcss/node";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HttpError } from "@openhands/typescript-client";
import { AxiosError } from "axios";
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
import { useEffect, useState } from "react";
import {
  PENDING_AUTOMATION_SETUP_ID,
  clearAutomationFormSession,
  getAutomationFormSession,
  initializeAutomationFormSession,
  subscribeAutomationFormSession,
} from "#/api/automation-form-session";
import type { AutomationSetupDraft } from "#/api/automation-setup-types";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { packTarGzip } from "#/utils/tar-gzip";
import { handleAutomationFormUpdateAction } from "#/services/automation-form";
import { AUTOMATION_FORM_UPDATE_ACTION_KIND } from "#/constants/automation-form";
import { AUTOMATION_SETUP_SHOW_AGENT_EVENT } from "#/components/features/automations/setup/automation-setup-agent-request";
import { setupDraftFromAutomation } from "#/utils/automation-edit-draft";
import { useDeploymentCapabilities } from "#/hooks/query/use-manifest-capabilities";
import { AUTOMATION_DETAIL_QUERY_KEY } from "#/hooks/query/use-automation-detail";
import { AUTOMATIONS_QUERY_KEY } from "#/hooks/query/use-automations";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import type { AutomationDraftApiResponse } from "#/manifests/types";
import type { Automation } from "#/types/automation";
import type { Provider } from "#/types/settings";

const mockNavigate = vi.fn();
const mockToastSuccess = vi.fn();
const createGitRepositoriesQueryResult = () => ({
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
        missing_token: false,
        next_page_id: null,
      },
    ],
  },
  isError: false,
  isLoading: false,
  hasNextPage: false,
  isFetchingNextPage: false,
  fetchNextPage: vi.fn(),
  onLoadMore: vi.fn(),
});
/** One private repository listed by another provider than GitHub. */
const createProviderRepositoriesQueryResult = (
  provider: Provider,
  fullName: string,
) => ({
  ...createGitRepositoriesQueryResult(),
  data: {
    pages: [
      {
        items: [
          {
            id: `${provider}-1`,
            full_name: fullName,
            git_provider: provider,
            is_public: false,
          },
        ],
        missing_token: false,
        next_page_id: null,
      },
    ],
  },
});
const mockUseGitRepositories = vi.fn(
  (_options?: { provider?: Provider | null }) =>
    createGitRepositoriesQueryResult(),
);
const mockUseSearchRepositories = vi.fn((..._args: unknown[]) => ({
  data: undefined as { full_name: string }[] | undefined,
  isFetching: false,
}));
const mockUseSettings = vi.fn(() => ({
  data: { provider_tokens_set: {} as Record<string, string | null> },
}));

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
      updateConversationTags: vi.fn(
        async (_conversationId: string, tags: Record<string, string>) => ({
          tags,
        }),
      ),
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
    dispatchAutomation: vi.fn(),
    updateAutomation: vi.fn(),
    getAutomation: vi.fn(),
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

const mockUseUserProviders = vi.fn(() => ({
  providers: ["github"],
  isLoadingSettings: false,
}));

vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: () => mockUseUserProviders(),
}));

vi.mock("#/hooks/query/use-git-repositories", () => ({
  useGitRepositories: (options: { provider?: Provider | null }) =>
    mockUseGitRepositories(options),
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => mockUseSettings(),
}));

vi.mock("#/hooks/query/use-search-repositories", () => ({
  useSearchRepositories: (...args: unknown[]) =>
    mockUseSearchRepositories(...args),
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
  automationListPath: () => "/automations",
  automationDetailPath: (id: string) => `/automations/${id}`,
  getAutomationEndpoint: (name: string) =>
    name === "createPlugin"
      ? "/v1/preset/plugin"
      : name === "createBundle"
        ? "/v1"
        : "/v1/preset/prompt",
}));

/** An axios error carrying a server response, as the Local transport throws. */
function axiosResponseError(status: number, data: unknown): AxiosError {
  const error = new AxiosError(`Request failed with status code ${status}`);
  error.response = { status, data } as AxiosError["response"];
  return error;
}

function renderPanel(
  draft: AutomationSetupDraft = {
    prompt: "Review every pull request",
    kind: "prompt",
  },
  conversationId = "conv-1",
  conversationTags: Record<string, string> | null | undefined = null,
  queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  }),
  onAskAgent?: (prompt: string) => void,
) {
  const value: NavigationContextValue = {
    currentPath: `/conversations/${conversationId}`,
    conversationId,
    isNavigating: false,
    navigate: mockNavigate,
  };

  return render(
    <QueryClientProvider client={queryClient}>
      <NavigationProvider value={value}>
        <AutomationSetupPanel
          draft={draft}
          conversationId={conversationId}
          conversationTags={conversationTags}
          onAskAgent={onAskAgent}
        />
      </NavigationProvider>
    </QueryClientProvider>,
  );
}

/** Feeds the panel from the form session on every change, as the setup route does. */
function SessionBoundPanel({ conversationId }: { conversationId: string }) {
  const [draft, setDraft] = useState(
    () => getAutomationFormSession(conversationId) as AutomationSetupDraft,
  );
  useEffect(
    () =>
      subscribeAutomationFormSession(conversationId, (next) => {
        if (next) setDraft(next);
      }),
    [conversationId],
  );
  return (
    <AutomationSetupPanel
      draft={draft}
      conversationId={conversationId}
      conversationTags={null}
    />
  );
}

function renderSessionBoundPanel(conversationId: string) {
  const value: NavigationContextValue = {
    currentPath: "/automations/setup",
    conversationId: null,
    isNavigating: false,
    navigate: mockNavigate,
  };
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <NavigationProvider value={value}>
        <SessionBoundPanel conversationId={conversationId} />
      </NavigationProvider>
    </QueryClientProvider>,
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
    ).mockImplementation(async (_conversationId, tags) => ({ tags }) as never);
    mockUseGitRepositories.mockReturnValue(createGitRepositoriesQueryResult());
    mockUseSearchRepositories.mockReturnValue({
      data: undefined,
      isFetching: false,
    });
    mockUseUserProviders.mockReturnValue({
      providers: ["github"],
      isLoadingSettings: false,
    });
    mockUseSettings.mockReturnValue({ data: { provider_tokens_set: {} } });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("preserves fixed card edges and root-scaled schedule fields in compiled CSS", async () => {
    // Arrange: use the real stylesheet and classes emitted by the panel.
    const compiler = await compile(
      await readFile(resolve("src/tailwind.css"), "utf8"),
      { base: resolve("src"), onDependency: () => {} },
    );
    const user = userEvent.setup();
    const { container } = renderPanel();
    const style = document.createElement("style");
    const originalRootFontSize = document.documentElement.style.fontSize;
    document.head.appendChild(style);
    const applyLayout = () => {
      const classes = new Set(
        [...container.querySelectorAll("[class]")].flatMap((element) => [
          ...element.classList,
        ]),
      );
      // jsdom does not apply cascade layers. Keep each matching compiled rule
      // in its original order, so shared form styles still participate.
      style.textContent = [
        ...compiler
          .build([...classes])
          .matchAll(/\.((?:\\.|[\w-])+)\s*\{([^{}]+)\}/g),
      ]
        .filter(([, selector]) => classes.has(selector.replace(/\\(.)/g, "$1")))
        .map(([rule]) => rule)
        .join("\n");
    };
    const assertJoinedCard = (card: HTMLElement, tray: HTMLElement) => {
      expect(getComputedStyle(card).borderRadius).toBe("15px");
      expect(getComputedStyle(card).marginBottom).toBe("-15px");
      const trayStyle = getComputedStyle(tray);
      expect(trayStyle.borderBottomLeftRadius).toBe("15px");
      expect(trayStyle.borderBottomRightRadius).toBe("15px");
      expect(trayStyle.paddingTop).toBe("23px");
    };
    const scheduleWidth = (element: HTMLElement) => {
      // jsdom keeps var() in computed widths. Resolve it from the actual row
      // to verify both the field's winning width and the variable's owner.
      const rowStyle = getComputedStyle(
        screen.getByTestId("automation-setup-at-row"),
      );
      return getComputedStyle(element).width.replace(
        /var\((--[\w-]+)\)/g,
        (_, property: string) => rowStyle.getPropertyValue(property).trim(),
      );
    };

    try {
      // Act / Assert: the prompt and custom-code modes share the joined edge,
      // while their editing areas retain their distinct minimum heights.
      applyLayout();
      assertJoinedCard(
        screen.getByTestId("automation-setup-prompt-container"),
        screen.getByTestId("automation-setup-prompt-drawer"),
      );
      expect(
        getComputedStyle(screen.getByTestId("automation-setup-prompt"))
          .minHeight,
      ).toBe("120px");
      expect(scheduleWidth(screen.getByTestId("automation-setup-time"))).toBe(
        "9.5rem",
      );

      // Host font-size changes scale the rem-based fields, while the joined
      // card edge stays fixed and the tray keeps its mixed px/rem inset.
      document.documentElement.style.fontSize = "20px";
      expect(
        scheduleWidth(
          screen.getByTestId("automation-setup-timezone").parentElement!,
        ),
      ).toBe("16rem");
      expect(
        getComputedStyle(screen.getByTestId("automation-setup-prompt-drawer"))
          .paddingTop,
      ).toBe("25px");
      expect(
        getComputedStyle(
          screen.getByTestId("automation-setup-prompt-container"),
        ).borderRadius,
      ).toBe("15px");

      await user.click(screen.getByTestId("automation-setup-frequency-once"));
      applyLayout();
      expect(
        scheduleWidth(screen.getByTestId("automation-setup-datetime")),
      ).toBe("15rem");
      await user.click(screen.getByTestId("automation-setup-frequency-custom"));
      applyLayout();
      expect(
        scheduleWidth(screen.getByTestId("automation-setup-custom-schedule")),
      ).toBe("14rem");

      document.documentElement.style.fontSize = originalRootFontSize;
      await user.click(screen.getByTestId("automation-setup-kind-custom"));
      applyLayout();
      const codeScroll = screen.getByTestId(
        "automation-setup-custom-code-scroll",
      );
      assertJoinedCard(
        codeScroll.parentElement!,
        screen.getByTestId("automation-setup-custom-details-drawer"),
      );
      expect(getComputedStyle(codeScroll).minHeight).toBe("256px");
      const grip = screen.getByTestId("automation-setup-custom-code-grip");
      expect(getComputedStyle(grip.children[0]).zIndex).toBe("1");
      expect(getComputedStyle(grip.children[1]).zIndex).toBe("2");
    } finally {
      document.documentElement.style.fontSize = originalRootFontSize;
      style.remove();
    }
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

  it("runs preflight validation without draft requests when draft capabilities are unsupported", async () => {
    vi.mocked(AutomationService.validateDraft).mockResolvedValue({
      valid: true,
      errors: [],
    });

    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-test"));

    expect(AutomationService.createServerDraft).not.toHaveBeenCalled();
    expect(AutomationService.updateServerDraft).not.toHaveBeenCalled();
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
    expect(mockToastSuccess).toHaveBeenCalledWith(
      "AUTOMATION_SETUP$READY_TO_TEST",
    );
    expect(
      screen.queryByTestId("automation-setup-status"),
    ).not.toBeInTheDocument();
  });

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

  it("does not offer or send an agent profile for prompt automations", async () => {
    vi.mocked(AutomationService.createServerDraft).mockRejectedValue({
      response: { status: 404 },
    });
    vi.mocked(AutomationService.validateDraft).mockResolvedValue({
      valid: true,
      errors: [],
    });

    const user = userEvent.setup();
    renderPanel({
      prompt: "Review every pull request",
      kind: "prompt",
      form: { kind: "prompt", agentProfileId: "agent-reviewer" },
    });

    expect(
      screen.queryByTestId("automation-setup-agent-profile"),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId("automation-setup-model")).toBeInTheDocument();

    await user.click(screen.getByTestId("automation-setup-test"));

    await waitFor(() =>
      expect(AutomationService.validateDraft).toHaveBeenCalled(),
    );
    expect(
      vi.mocked(AutomationService.validateDraft).mock.calls[0][0].draft,
    ).not.toHaveProperty("agent_profile_id");
  });

  it("leaves a prompt automation's stored agent profile alone on save", async () => {
    vi.mocked(AutomationService.updateAutomation).mockResolvedValue({
      id: "auto-1",
    } as never);

    const user = userEvent.setup();
    renderPanel({
      prompt: "Review every pull request",
      kind: "prompt",
      editingAutomationId: "auto-1",
      form: {
        kind: "prompt",
        name: "Daily digest",
        prompt: "Review every pull request",
        agentProfileId: "agent-reviewer",
      },
    });

    await user.click(screen.getByTestId("automation-setup-save"));

    await waitFor(() =>
      expect(AutomationService.updateAutomation).toHaveBeenCalledTimes(1),
    );
    expect(
      vi.mocked(AutomationService.updateAutomation).mock.calls[0][1],
    ).not.toHaveProperty("agent_profile_id");
  });

  it("closes the model menu on Escape and returns focus to its trigger", async () => {
    const user = userEvent.setup();
    renderPanel();

    const modelTrigger = screen.getByTestId("automation-setup-model");
    await user.click(modelTrigger);
    expect(
      screen.getByTestId("automation-setup-model-popover"),
    ).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(
      screen.queryByTestId("automation-setup-model-popover"),
    ).not.toBeInTheDocument();
    expect(modelTrigger).toHaveAttribute("aria-expanded", "false");
    expect(modelTrigger).toHaveFocus();
  });

  it("closes the repository menu on Escape, also from its search field", async () => {
    const user = userEvent.setup();
    renderPanel();

    const repositoryTrigger = screen.getByTestId(
      "automation-setup-repository-add",
    );
    await user.click(repositoryTrigger);
    expect(
      screen.getByTestId("automation-setup-repository-menu"),
    ).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(
      screen.queryByTestId("automation-setup-repository-menu"),
    ).not.toBeInTheDocument();
    expect(repositoryTrigger).toHaveFocus();

    // Escape typed in the menu's search field closes the menu too, while
    // other keys stay in the field.
    await user.click(repositoryTrigger);
    const search = screen.getByTestId("automation-setup-repository-search");
    await user.type(search, "qa");
    expect(search).toHaveValue("qa");
    await user.keyboard("{Escape}");
    expect(
      screen.queryByTestId("automation-setup-repository-menu"),
    ).not.toBeInTheDocument();
    expect(repositoryTrigger).toHaveFocus();
  });

  it("closes the event source menu on Escape and returns focus to its toggle", async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByText("AUTOMATIONS$DETAIL$TRIGGER_EVENT"));
    const sourceToggle = screen.getByTestId(
      "automation-setup-event-source-toggle",
    );
    await user.click(sourceToggle);
    expect(
      screen.getByTestId("automation-setup-event-source-menu"),
    ).toBeInTheDocument();
    expect(sourceToggle).toHaveAttribute("aria-expanded", "true");
    await user.keyboard("{Escape}");
    expect(
      screen.queryByTestId("automation-setup-event-source-menu"),
    ).not.toBeInTheDocument();
    expect(sourceToggle).toHaveAttribute("aria-expanded", "false");
    expect(sourceToggle).toHaveFocus();
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
    expect(
      screen.getByTestId("automation-setup-repository-values"),
    ).toHaveClass("flex-wrap");
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
    await user.click(screen.getByTestId("automation-setup-repository-add"));
    await user.click(screen.getByTestId("automation-setup-repository-custom"));
    await user.type(
      screen.getByTestId("automation-setup-repository-address"),
      "https://gitlab.com/acme/tools",
    );
    await user.click(screen.getByTestId("automation-setup-repository-submit"));
    await user.click(screen.getByTestId("automation-setup-test"));

    // A full URL names its own host, so it is cloned without a GitHub token.
    await waitFor(() =>
      expect(AutomationService.validateDraft).toHaveBeenCalledWith(
        expect.objectContaining({
          draft: expect.objectContaining({
            repos: [
              { url: "OpenHands/OpenHands", provider: "github" },
              { url: "OpenHands/software-agent-sdk", provider: "github" },
              { url: "https://gitlab.com/acme/tools" },
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

    const option = await screen.findByTestId(
      "automation-setup-repository-option-OpenHands/OpenHands",
    );
    // Long organization/project/repository names are truncated, so the full
    // name is available on hover.
    expect(within(option).getByTitle("OpenHands/OpenHands")).toBeVisible();
    await user.click(option);
    const pill = screen.getByTestId("automation-setup-repository-value");
    expect(pill).toHaveTextContent("OpenHands/OpenHands");
    expect(within(pill).getByTitle("OpenHands/OpenHands")).toBeVisible();

    await user.click(screen.getByTestId("automation-setup-repository-add"));
    await user.click(screen.getByTestId("automation-setup-repository-custom"));
    expect(
      screen.getByTestId("automation-setup-add-repository-modal"),
    ).toBeInTheDocument();
  });

  it("shows a repository loading error in the repository picker", async () => {
    mockUseGitRepositories.mockReturnValue({
      data: { pages: [] },
      isError: true,
      isLoading: false,
      hasNextPage: false,
      isFetchingNextPage: false,
      fetchNextPage: vi.fn(),
      onLoadMore: vi.fn(),
    });

    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-repository-add"));

    expect(
      screen.getByTestId("automation-setup-repository-error"),
    ).toHaveTextContent("HOME$FAILED_TO_LOAD_REPOSITORIES");
    expect(
      screen.queryByTestId("automation-setup-repository-loading"),
    ).not.toBeInTheDocument();
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

  it("searches the provider for repositories that are not loaded yet", async () => {
    mockUseSearchRepositories.mockImplementation((query) => ({
      data:
        query === "agent"
          ? [{ full_name: "OpenHands/agent-canvas-extras" }]
          : undefined,
      isFetching: false,
    }));

    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-repository-add"));
    await user.type(
      screen.getByTestId("automation-setup-repository-search"),
      "agent",
    );

    expect(
      await screen.findByTestId(
        "automation-setup-repository-option-OpenHands/agent-canvas-extras",
      ),
    ).toBeInTheDocument();
    // A loaded repository that matches stays listed.
    expect(
      screen.getByTestId(
        "automation-setup-repository-option-OpenHands/software-agent-sdk",
      ),
    ).toBeInTheDocument();
    expect(mockUseSearchRepositories).toHaveBeenLastCalledWith(
      "agent",
      "github",
      false,
    );
  });

  it("explains that automations cannot clone another provider's repositories", async () => {
    mockUseUserProviders.mockReturnValue({
      providers: ["azure_devops"],
      isLoadingSettings: false,
    });
    mockUseGitRepositories.mockReturnValue({
      ...createGitRepositoriesQueryResult(),
      data: {
        pages: [
          {
            items: [
              {
                id: "ado-1",
                full_name: "fabrikam/Fabrikam Fiber/fabrikam-web",
                git_provider: "azure_devops",
                is_public: false,
              },
            ],
            missing_token: false,
            next_page_id: null,
          },
        ],
      },
    });

    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-repository-add"));

    expect(
      screen.getByTestId("automation-setup-repository-unsupported"),
    ).toHaveTextContent("AUTOMATION_SETUP$REPOSITORY_PROVIDER_UNSUPPORTED");
    const option = screen.getByTestId(
      "automation-setup-repository-option-fabrikam/Fabrikam Fiber/fabrikam-web",
    );
    expect(option).toBeDisabled();
    await user.click(option);
    expect(
      screen.queryByTestId("automation-setup-repository-value"),
    ).not.toBeInTheDocument();
  });

  it("adds a GitLab repository by its URL and sends it without a provider", async () => {
    vi.mocked(AutomationService.createServerDraft).mockRejectedValue({
      response: { status: 404 },
    });
    vi.mocked(AutomationService.validateDraft).mockResolvedValue({
      valid: true,
      errors: [],
    });
    mockUseUserProviders.mockReturnValue({
      providers: ["gitlab"],
      isLoadingSettings: false,
    });
    // The server names the connection's host; gitlab.com is the hosted service.
    mockUseSettings.mockReturnValue({
      data: { provider_tokens_set: { gitlab: "gitlab.com" } },
    });
    mockUseGitRepositories.mockReturnValue(
      createProviderRepositoriesQueryResult("gitlab", "acme/tools"),
    );

    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-repository-add"));
    // One cloneable provider needs no provider choice.
    expect(
      screen.queryByTestId("automation-setup-repository-provider"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-setup-repository-unsupported"),
    ).not.toBeInTheDocument();
    await user.click(
      screen.getByTestId("automation-setup-repository-option-acme/tools"),
    );

    expect(
      screen.getByTestId("automation-setup-repository-value"),
    ).toHaveTextContent("https://gitlab.com/acme/tools");

    await user.click(screen.getByTestId("automation-setup-test"));

    // gitlab.com is a hosted service the run recognizes on its own.
    await waitFor(() =>
      expect(AutomationService.validateDraft).toHaveBeenCalledWith(
        expect.objectContaining({
          draft: expect.objectContaining({
            repos: [{ url: "https://gitlab.com/acme/tools" }],
          }),
        }),
      ),
    );
  });

  it("keeps GitLab repositories disabled when the server reports no host", async () => {
    // The same name covers gitlab.com and a self-managed instance, so without
    // the host the form cannot address the repository and must not guess.
    mockUseUserProviders.mockReturnValue({
      providers: ["gitlab"],
      isLoadingSettings: false,
    });
    mockUseSettings.mockReturnValue({
      data: { provider_tokens_set: { gitlab: null } },
    });
    mockUseGitRepositories.mockReturnValue(
      createProviderRepositoriesQueryResult("gitlab", "acme/tools"),
    );

    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-repository-add"));

    expect(
      screen.queryByTestId("automation-setup-repository-provider"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("automation-setup-repository-unsupported"),
    ).toHaveTextContent("AUTOMATION_SETUP$REPOSITORY_PROVIDER_UNSUPPORTED");
    const option = screen.getByTestId(
      "automation-setup-repository-option-acme/tools",
    );
    expect(option).toBeDisabled();
    await user.click(option);
    expect(
      screen.queryByTestId("automation-setup-repository-value"),
    ).not.toBeInTheDocument();
  });

  it("names the provider of a repository on a connected self-hosted host", async () => {
    vi.mocked(AutomationService.createServerDraft).mockRejectedValue({
      response: { status: 404 },
    });
    vi.mocked(AutomationService.validateDraft).mockResolvedValue({
      valid: true,
      errors: [],
    });
    mockUseUserProviders.mockReturnValue({
      providers: ["gitlab"],
      isLoadingSettings: false,
    });
    mockUseSettings.mockReturnValue({
      data: { provider_tokens_set: { gitlab: "https://gitlab.example.com" } },
    });
    mockUseGitRepositories.mockReturnValue(
      createProviderRepositoriesQueryResult("gitlab", "acme/tools"),
    );

    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-repository-add"));
    await user.click(
      screen.getByTestId("automation-setup-repository-option-acme/tools"),
    );

    expect(
      screen.getByTestId("automation-setup-repository-value"),
    ).toHaveTextContent("https://gitlab.example.com/acme/tools");

    await user.click(screen.getByTestId("automation-setup-test"));

    // The run cannot tell whose host this is, so the provider goes along.
    await waitFor(() =>
      expect(AutomationService.validateDraft).toHaveBeenCalledWith(
        expect.objectContaining({
          draft: expect.objectContaining({
            repos: [
              {
                url: "https://gitlab.example.com/acme/tools",
                provider: "gitlab",
              },
            ],
          }),
        }),
      ),
    );
  });

  it("sends no provider for a pasted URL on a host the run cannot clone from", async () => {
    vi.mocked(AutomationService.createServerDraft).mockRejectedValue({
      response: { status: 404 },
    });
    vi.mocked(AutomationService.validateDraft).mockResolvedValue({
      valid: true,
      errors: [],
    });
    mockUseUserProviders.mockReturnValue({
      providers: ["github", "bitbucket_data_center"],
      isLoadingSettings: false,
    });
    mockUseSettings.mockReturnValue({
      data: {
        provider_tokens_set: {
          github: "",
          bitbucket_data_center: "https://bb.example.com",
        },
      },
    });

    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-repository-add"));
    // Only GitHub is cloneable here, so there is no provider choice to make.
    expect(
      screen.queryByTestId("automation-setup-repository-provider"),
    ).not.toBeInTheDocument();
    await user.click(screen.getByTestId("automation-setup-repository-custom"));
    await user.type(
      screen.getByTestId("automation-setup-repository-address"),
      "https://bb.example.com/PROJ/repo",
    );
    await user.click(screen.getByTestId("automation-setup-repository-submit"));
    await user.click(screen.getByTestId("automation-setup-test"));

    // The service's schema does not know bitbucket_data_center, so the URL
    // goes alone rather than failing the whole request.
    await waitFor(() =>
      expect(AutomationService.validateDraft).toHaveBeenCalledWith(
        expect.objectContaining({
          draft: expect.objectContaining({
            repos: [{ url: "https://bb.example.com/PROJ/repo" }],
          }),
        }),
      ),
    );
  });

  it("lets a user with several providers choose whose repositories to list", async () => {
    mockUseUserProviders.mockReturnValue({
      providers: ["github", "gitlab"],
      isLoadingSettings: false,
    });
    mockUseSettings.mockReturnValue({
      data: { provider_tokens_set: { github: null, gitlab: "gitlab.com" } },
    });
    mockUseGitRepositories.mockImplementation((options) =>
      options?.provider === "gitlab"
        ? createProviderRepositoriesQueryResult("gitlab", "acme/tools")
        : createGitRepositoriesQueryResult(),
    );

    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-repository-add"));
    const providerSelect = screen.getByTestId(
      "automation-setup-repository-provider",
    );
    expect(providerSelect).toHaveValue("github");
    expect(
      within(providerSelect)
        .getAllByRole("option")
        .map((option) => option.textContent),
    ).toEqual(["GitHub", "GitLab"]);
    expect(
      screen.getByTestId(
        "automation-setup-repository-option-OpenHands/OpenHands",
      ),
    ).toBeInTheDocument();

    await user.selectOptions(providerSelect, "gitlab");

    expect(
      await screen.findByTestId(
        "automation-setup-repository-option-acme/tools",
      ),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId(
        "automation-setup-repository-option-OpenHands/OpenHands",
      ),
    ).not.toBeInTheDocument();
    expect(mockUseSearchRepositories).toHaveBeenLastCalledWith(
      "",
      "gitlab",
      false,
    );

    await user.click(
      screen.getByTestId("automation-setup-repository-option-acme/tools"),
    );
    expect(
      screen.getByTestId("automation-setup-repository-value"),
    ).toHaveTextContent("https://gitlab.com/acme/tools");
  });

  it("offers Azure DevOps repositories when the service clones them", async () => {
    vi.mocked(useDeploymentCapabilities).mockReturnValue({
      data: {
        ready: true,
        features: [],
        triggerKinds: ["cron"],
        eventSources: [],
        eventTypes: [],
        triggers: {},
        repoProviders: ["azure_devops", "github"],
      },
      isLoading: false,
    } as never);
    vi.mocked(AutomationService.createServerDraft).mockRejectedValue({
      response: { status: 404 },
    });
    vi.mocked(AutomationService.validateDraft).mockResolvedValue({
      valid: true,
      errors: [],
    });
    mockUseUserProviders.mockReturnValue({
      providers: ["azure_devops"],
      isLoadingSettings: false,
    });
    mockUseGitRepositories.mockReturnValue(
      createProviderRepositoriesQueryResult(
        "azure_devops",
        "fabrikam/Fabrikam Fiber/fabrikam-web",
      ),
    );

    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-repository-add"));
    expect(
      screen.queryByTestId("automation-setup-repository-unsupported"),
    ).not.toBeInTheDocument();
    const option = screen.getByTestId(
      "automation-setup-repository-option-fabrikam/Fabrikam Fiber/fabrikam-web",
    );
    expect(option).not.toBeDisabled();
    await user.click(option);

    expect(
      screen.getByTestId("automation-setup-repository-value"),
    ).toHaveTextContent(
      "https://dev.azure.com/fabrikam/Fabrikam%20Fiber/_git/fabrikam-web",
    );

    await user.click(screen.getByTestId("automation-setup-test"));

    await waitFor(() =>
      expect(AutomationService.validateDraft).toHaveBeenCalledWith(
        expect.objectContaining({
          draft: expect.objectContaining({
            repos: [
              {
                url: "https://dev.azure.com/fabrikam/Fabrikam%20Fiber/_git/fabrikam-web",
              },
            ],
          }),
        }),
      ),
    );
  });

  it("offers Bitbucket Data Center repositories when the service clones them", async () => {
    vi.mocked(useDeploymentCapabilities).mockReturnValue({
      data: {
        ready: true,
        features: [],
        triggerKinds: ["cron"],
        eventSources: [],
        eventTypes: [],
        triggers: {},
        repoProviders: ["bitbucket_data_center", "github"],
      },
      isLoading: false,
    } as never);
    vi.mocked(AutomationService.createServerDraft).mockRejectedValue({
      response: { status: 404 },
    });
    vi.mocked(AutomationService.validateDraft).mockResolvedValue({
      valid: true,
      errors: [],
    });
    mockUseUserProviders.mockReturnValue({
      providers: ["bitbucket_data_center"],
      isLoadingSettings: false,
    });
    mockUseSettings.mockReturnValue({
      data: {
        provider_tokens_set: {
          bitbucket_data_center: "bitbucket.example.com",
        },
      },
    });
    mockUseGitRepositories.mockReturnValue(
      createProviderRepositoriesQueryResult(
        "bitbucket_data_center",
        "DEM/demo-app",
      ),
    );

    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-repository-add"));
    const option = screen.getByTestId(
      "automation-setup-repository-option-DEM/demo-app",
    );
    expect(option).not.toBeDisabled();
    await user.click(option);

    expect(
      screen.getByTestId("automation-setup-repository-value"),
    ).toHaveTextContent("https://bitbucket.example.com/scm/DEM/demo-app.git");

    await user.click(screen.getByTestId("automation-setup-test"));

    await waitFor(() =>
      expect(AutomationService.validateDraft).toHaveBeenCalledWith(
        expect.objectContaining({
          draft: expect.objectContaining({
            repos: [
              {
                url: "https://bitbucket.example.com/scm/DEM/demo-app.git",
                provider: "bitbucket_data_center",
              },
            ],
          }),
        }),
      ),
    );
  });

  it("streams agent form updates field by field without overwriting user edits", async () => {
    vi.useFakeTimers();
    const conversationId = "conv-agent-updates";
    const draft: AutomationSetupDraft = {
      prompt: "Review every pull request",
      kind: "prompt",
    };
    initializeAutomationFormSession(conversationId, draft);
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
    expect(
      screen.getAllByText("AUTOMATION_SETUP$FILLED_BY_OPENHANDS").length,
    ).toBeGreaterThan(0);
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

  it("starts a Custom schedule from the preset it replaces", async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-frequency-weekly"));
    await user.selectOptions(
      screen.getByTestId("automation-setup-weekday"),
      "3",
    );
    fireEvent.change(screen.getByTestId("automation-setup-time"), {
      target: { value: "10:30" },
    });
    await user.click(screen.getByTestId("automation-setup-frequency-custom"));

    const custom = screen.getByTestId("automation-setup-custom-schedule");
    expect(custom).toHaveValue("30 10 * * 3");

    // Re-selecting Custom keeps what the user typed.
    fireEvent.change(custom, { target: { value: "15 8 * * 1-5" } });
    await user.click(screen.getByTestId("automation-setup-frequency-custom"));
    expect(screen.getByTestId("automation-setup-custom-schedule")).toHaveValue(
      "15 8 * * 1-5",
    );

    // Leaving for a preset and coming back starts from that preset again.
    await user.click(screen.getByTestId("automation-setup-frequency-daily"));
    await user.click(screen.getByTestId("automation-setup-frequency-custom"));
    expect(screen.getByTestId("automation-setup-custom-schedule")).toHaveValue(
      "30 10 * * *",
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
    expect(
      screen.getByTestId("automation-setup-add-event-filter"),
    ).toHaveTextContent("AUTOMATION_SETUP$ADD_FILTER");

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

  it("starts the setup conversation from Ask Agent when the panel has none", async () => {
    const user = userEvent.setup();
    const onAskAgent = vi.fn();
    const showAgent = vi.fn();
    window.addEventListener(AUTOMATION_SETUP_SHOW_AGENT_EVENT, showAgent);

    renderPanel(
      { prompt: "Review every pull request", kind: "prompt" },
      PENDING_AUTOMATION_SETUP_ID,
      null,
      undefined,
      onAskAgent,
    );

    await user.click(screen.getByText("AUTOMATIONS$DETAIL$TRIGGER_EVENT"));
    await user.click(screen.getByTestId("automation-setup-ask-agent"));

    expect(onAskAgent).toHaveBeenCalledWith(
      "AUTOMATION_SETUP$ASK_AGENT_EVENT_PROMPT",
    );
    // There is no conversation to send into yet.
    expect(mockSendMessage).not.toHaveBeenCalled();
    expect(showAgent).not.toHaveBeenCalled();
    window.removeEventListener(AUTOMATION_SETUP_SHOW_AGENT_EVENT, showAgent);
  });

  it("shows a spinner in the repository menu while the list loads", async () => {
    mockUseGitRepositories.mockReturnValue({
      data: { pages: [] },
      isError: false,
      isLoading: true,
      hasNextPage: false,
      isFetchingNextPage: false,
      fetchNextPage: vi.fn(),
      onLoadMore: vi.fn(),
    });
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-repository-add"));

    expect(
      screen.getByTestId("automation-setup-repository-loading"),
    ).toBeInTheDocument();
  });

  it("shows a settings secret tooltip when local repository token is missing", async () => {
    mockUseGitRepositories.mockReturnValue({
      data: {
        pages: [
          {
            items: [],
            missing_token: true,
            next_page_id: null,
          },
        ],
      },
      isError: false,
      isLoading: false,
      hasNextPage: false,
      isFetchingNextPage: false,
      fetchNextPage: vi.fn(),
      onLoadMore: vi.fn(),
    });
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-repository-add"));

    expect(
      await screen.findByTestId("automation-setup-repository-token-tooltip"),
    ).toHaveAccessibleName("AUTOMATION_SETUP$GITHUB_TOKEN_REQUIRED_TOOLTIP");
    expect(
      screen.getByTestId("automation-setup-repository-custom"),
    ).toBeInTheDocument();
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
      screen
        .getByTestId("automation-setup-plugin-ref")
        .compareDocumentPosition(
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

  it("creates custom bundle drafts with entrypoint, setup script path and model", async () => {
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
    await user.click(screen.getByTestId("automation-setup-model"));
    await user.click(screen.getByTestId("automation-setup-model-option-fast"));
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
        model: "fast",
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

    const echoSavedDraft = (
      request: {
        endpoint?: AutomationDraftApiResponse["endpoint"];
        name?: string | null;
        draft?: AutomationDraftApiResponse["draft"];
      },
      overrides: Partial<AutomationDraftApiResponse> = {},
    ): AutomationDraftApiResponse => ({
      ...dispatchableDraft,
      endpoint: request.endpoint ?? dispatchableDraft.endpoint,
      name: request.name ?? null,
      draft: request.draft ?? dispatchableDraft.draft,
      ...overrides,
    });

    const mockSavedDraftEcho = (
      overrides: Partial<AutomationDraftApiResponse> = {},
    ) => {
      vi.mocked(AutomationService.createServerDraft).mockImplementation(
        async (request) => echoSavedDraft(request, overrides),
      );
      vi.mocked(AutomationService.updateServerDraft).mockImplementation(
        async (_id, request) => echoSavedDraft(request, overrides),
      );
    };

    const saveDraftAndWaitForTest = async (
      user: ReturnType<typeof userEvent.setup>,
    ) => {
      await user.click(screen.getByTestId("automation-setup-save-draft"));
      await waitFor(() =>
        expect(screen.getByTestId("automation-setup-test")).toBeEnabled(),
      );
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

    it("does not overwrite tags before conversation tags have loaded", async () => {
      const value: NavigationContextValue = {
        currentPath: "/conversations/conv-1",
        conversationId: "conv-1",
        isNavigating: false,
        navigate: mockNavigate,
      };
      const queryClient = new QueryClient({
        defaultOptions: {
          queries: { retry: false },
          mutations: { retry: false },
        },
      });

      render(
        <QueryClientProvider client={queryClient}>
          <NavigationProvider value={value}>
            <AutomationSetupPanel
              draft={{ prompt: "Edit automation", kind: "prompt" }}
              conversationId="conv-1"
            />
          </NavigationProvider>
        </QueryClientProvider>,
      );

      await act(async () => {
        await Promise.resolve();
      });

      expect(
        AgentServerConversationService.updateConversationTags,
      ).not.toHaveBeenCalled();
    });

    it("omits an empty prompt when saving a partial prompt draft", async () => {
      vi.mocked(AutomationService.createServerDraft).mockResolvedValue({
        ...dispatchableDraft,
        draft: { name: "New Automation" } as never,
        validationErrors: null,
      });

      const user = userEvent.setup();

      renderPanel({ prompt: "", kind: "prompt" });

      await user.click(screen.getByTestId("automation-setup-save-draft"));

      await waitFor(() =>
        expect(AutomationService.createServerDraft).toHaveBeenCalledWith(
          expect.objectContaining({
            endpoint: "/v1/preset/prompt",
            draft: expect.not.objectContaining({ prompt: expect.any(String) }),
          }),
        ),
      );
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
      expect(mockToastSuccess).toHaveBeenCalledWith(
        "AUTOMATION_SETUP$DRAFT_SAVED",
      );
      expect(
        screen.queryByTestId("automation-setup-status"),
      ).not.toBeInTheDocument();
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

    it("uploads fresh custom bundles and keeps the saved draft clean", async () => {
      vi.mocked(AutomationService.uploadAutomationTarball)
        .mockResolvedValueOnce("oh-internal://uploads/custom-draft-1")
        .mockResolvedValueOnce("oh-internal://uploads/custom-draft-2");
      mockSavedDraftEcho({ validationErrors: null });

      const user = userEvent.setup();
      renderPanel({
        prompt: "Run generated Python",
        kind: "custom",
      });

      await user.click(screen.getByTestId("automation-setup-save-draft"));

      await waitFor(() =>
        expect(AutomationService.createServerDraft).toHaveBeenCalledWith(
          expect.objectContaining({
            endpoint: "/v1",
            draft: expect.objectContaining({
              tarball_path: "oh-internal://uploads/custom-draft-1",
              entrypoint: "python3 main.py",
              setup_script_path: "setup.sh",
            }),
          }),
        ),
      );
      await waitFor(() =>
        expect(
          screen.getByTestId("automation-setup-save-state"),
        ).toHaveTextContent("AUTOMATION_SETUP$SAVED_JUST_NOW"),
      );
      expect(
        screen.getByTestId("automation-setup-save-state"),
      ).not.toHaveTextContent("AUTOMATION_SETUP$UNSAVED_CHANGES");
      expect(AutomationService.uploadAutomationTarball).toHaveBeenCalledTimes(
        1,
      );
      expect(packTarGzip).toHaveBeenCalledWith([
        expect.objectContaining({ name: "main.py", mode: 0o644 }),
        expect.objectContaining({ name: "setup.sh", mode: 0o755 }),
      ]);

      await user.click(screen.getByTestId("automation-setup-save-draft"));

      await waitFor(() =>
        expect(AutomationService.updateServerDraft).toHaveBeenCalledWith(
          "draft-1",
          expect.objectContaining({
            draft: expect.objectContaining({
              tarball_path: "oh-internal://uploads/custom-draft-2",
            }),
          }),
        ),
      );
      expect(AutomationService.uploadAutomationTarball).toHaveBeenCalledTimes(
        2,
      );
    });

    it("opens test runs from the draft status and returns to the form", async () => {
      mockSavedDraftEcho({ validationErrors: null });
      vi.mocked(AutomationService.getServerDraft).mockResolvedValue(
        dispatchableDraft,
      );

      const user = userEvent.setup();
      renderPanel();

      expect(screen.getByTestId("automation-setup-test")).toBeDisabled();

      await saveDraftAndWaitForTest(user);
      expect(
        screen.getByTestId("automation-setup-draft-details"),
      ).toBeInTheDocument();
      expect(screen.getByTestId("automation-setup-draft-test")).toBeEnabled();

      await user.click(screen.getByTestId("automation-setup-view-test-runs"));

      expect(
        screen.getByTestId("automation-setup-draft-runs-page"),
      ).toBeInTheDocument();
      expect(
        screen.getByTestId("automation-setup-draft-runs-empty"),
      ).toHaveTextContent("AUTOMATIONS$DETAIL$NO_RUNS");
      expect(screen.getByTestId("automation-setup-form")).toHaveClass("hidden");

      await user.click(screen.getByTestId("automation-setup-draft-runs-back"));

      expect(screen.getByTestId("automation-setup-form")).not.toHaveClass(
        "hidden",
      );
      expect(
        screen.queryByTestId("automation-setup-draft-runs-page"),
      ).not.toBeInTheDocument();
    });

    it("shows draft and test errors on the test runs page", async () => {
      const blockedDraft = {
        ...dispatchableDraft,
        dispatchable: false,
        validationErrors: [
          {
            field: null,
            code: "webhook",
            message: "No webhook is configured",
          },
        ],
      };
      vi.mocked(AutomationService.createServerDraft).mockResolvedValue(
        blockedDraft,
      );
      vi.mocked(AutomationService.updateServerDraft).mockResolvedValue(
        blockedDraft,
      );

      const user = userEvent.setup();
      renderPanel();
      await user.click(screen.getByTestId("automation-setup-save-draft"));
      await screen.findByTestId("automation-setup-draft-details");
      await user.click(screen.getByTestId("automation-setup-view-test-runs"));

      expect(
        screen.getByTestId("automation-setup-draft-runs-error"),
      ).toHaveTextContent("No webhook is configured");

      const pageTest = screen
        .getAllByTestId("automation-setup-test")
        .find((button) =>
          screen
            .getByTestId("automation-setup-draft-runs-empty")
            .contains(button),
        );
      await user.click(pageTest!);

      await waitFor(() =>
        expect(
          screen.getAllByTestId("automation-setup-draft-runs-error"),
        ).toHaveLength(1),
      );
      expect(AutomationService.dispatchServerDraft).not.toHaveBeenCalled();
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

    it("stores the server draft id in the pending setup session after saving", async () => {
      vi.mocked(AutomationService.createServerDraft).mockResolvedValue(
        dispatchableDraft,
      );

      const user = userEvent.setup();
      renderPanel(undefined, PENDING_AUTOMATION_SETUP_ID, null);

      await user.click(screen.getByTestId("automation-setup-save-draft"));

      await waitFor(() =>
        expect(getAutomationFormSession(PENDING_AUTOMATION_SETUP_ID)).toEqual(
          expect.objectContaining({
            serverDraftId: "draft-1",
            form: expect.objectContaining({
              prompt: "Review every pull request",
            }),
          }),
        ),
      );
      expect(
        AgentServerConversationService.updateConversationTags,
      ).not.toHaveBeenCalled();
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
      expect(getAutomationFormSession("conv-1")).toEqual(
        expect.objectContaining({
          serverDraftId: "draft-1",
          form: expect.objectContaining({
            prompt: "Use the persisted draft body",
          }),
        }),
      );
    });

    it("does not let the setup-mode tag write clobber a saved draft id", async () => {
      vi.mocked(AutomationService.createServerDraft).mockResolvedValue(
        dispatchableDraft,
      );
      let resolveSetupTagWrite: (value: {
        tags: Record<string, string>;
      }) => void;
      const setupTagWrite = new Promise<{ tags: Record<string, string> }>(
        (resolve) => {
          resolveSetupTagWrite = resolve;
        },
      );
      vi.mocked(AgentServerConversationService.updateConversationTags)
        .mockImplementationOnce(() => setupTagWrite as never)
        .mockImplementationOnce(
          async (_conversationId, tags) => ({ tags }) as never,
        );

      const user = userEvent.setup();
      renderPanel(undefined, "conv-1", { existing: "tag" });

      await waitFor(() =>
        expect(
          AgentServerConversationService.updateConversationTags,
        ).toHaveBeenCalledTimes(1),
      );

      await user.click(screen.getByTestId("automation-setup-save-draft"));
      await waitFor(() =>
        expect(AutomationService.createServerDraft).toHaveBeenCalled(),
      );
      expect(
        AgentServerConversationService.updateConversationTags,
      ).toHaveBeenCalledTimes(1);

      resolveSetupTagWrite!({
        tags: { existing: "tag", automationsetup: "draft" },
      });

      await waitFor(() =>
        expect(
          AgentServerConversationService.updateConversationTags,
        ).toHaveBeenCalledTimes(2),
      );
      expect(
        vi.mocked(AgentServerConversationService.updateConversationTags).mock
          .calls[1][1],
      ).toEqual(
        expect.objectContaining({
          existing: "tag",
          automationsetup: "draft",
          automationdraftid: "draft-1",
        }),
      );
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
      mockSavedDraftEcho();
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

      await saveDraftAndWaitForTest(user);
      await user.click(screen.getByTestId("automation-setup-test"));

      await waitFor(() =>
        expect(AutomationService.dispatchServerDraft).toHaveBeenCalledWith(
          "draft-1",
        ),
      );
      expect(AutomationService.validateDraft).not.toHaveBeenCalled();
      expect(mockToastSuccess).toHaveBeenCalledWith(
        "AUTOMATION_SETUP$TEST_DISPATCHED",
      );
      expect(
        screen.queryByTestId("automation-setup-status"),
      ).not.toBeInTheDocument();
      expect(mockNavigate).not.toHaveBeenCalledWith(
        "/conversations/conv-run-1",
      );
    });

    it("surfaces validation errors when the draft is not dispatchable", async () => {
      const blocked = (
        request: Parameters<typeof echoSavedDraft>[0],
      ): AutomationDraftApiResponse =>
        echoSavedDraft(request, {
          dispatchable: false,
          validationErrors: [
            {
              field: "trigger.schedule",
              code: "interval_too_short",
              message: "Minimum interval is 5 minutes.",
            },
          ],
        });
      vi.mocked(AutomationService.createServerDraft).mockImplementation(
        async (request) => blocked(request),
      );
      vi.mocked(AutomationService.updateServerDraft).mockImplementation(
        async (_id, request) => blocked(request),
      );

      const user = userEvent.setup();
      renderPanel();

      await saveDraftAndWaitForTest(user);
      const validity = screen.getByTestId("automation-setup-draft-validity");
      expect(validity.parentElement).toBe(
        screen.getByTestId("automation-setup-draft-details"),
      );
      expect(validity.previousElementSibling).toContainElement(
        screen.getByTestId("automation-setup-draft-test"),
      );
      await user.click(screen.getByTestId("automation-setup-test"));

      await waitFor(() =>
        expect(
          screen.getByTestId("automation-setup-draft-runs-error"),
        ).toHaveTextContent("Minimum interval is 5 minutes."),
      );
      expect(
        screen.queryByTestId("automation-setup-status"),
      ).not.toBeInTheDocument();
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
      mockSavedDraftEcho();
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
      await saveDraftAndWaitForTest(user);
      await user.click(screen.getByTestId("automation-setup-test"));

      await waitFor(() =>
        expect(AutomationService.dispatchServerDraft).toHaveBeenCalledWith(
          "draft-1",
          { eventPayload: { type: "issue.created", action: "opened" } },
        ),
      );
    });

    it("treats the service's built-in sources as known before capabilities answer", async () => {
      // The default capabilities mock returns no data, so the panel is on its
      // fallback source list.
      const user = userEvent.setup();
      renderPanel();

      await user.click(screen.getByText("AUTOMATIONS$DETAIL$TRIGGER_EVENT"));
      await user.click(
        screen.getByTestId("automation-setup-event-source-toggle"),
      );
      await user.click(
        screen.getByTestId("automation-setup-event-source-custom"),
      );
      const sourceInput = screen.getByTestId("automation-setup-event-source");

      for (const builtin of [
        "jira_dc",
        "bitbucket_data_center",
        "azure_devops",
      ]) {
        fireEvent.change(sourceInput, { target: { value: builtin } });
        expect(
          screen.queryByTestId("automation-setup-custom-webhook-enabled"),
        ).not.toBeInTheDocument();
      }

      for (const external of ["linear", "slack"]) {
        fireEvent.change(sourceInput, { target: { value: external } });
        expect(
          screen.getByTestId("automation-setup-custom-webhook-enabled"),
        ).toBeChecked();
      }
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
      mockSavedDraftEcho();
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
      const sourceInput = screen.getByTestId("automation-setup-event-source");
      expect(sourceInput).toHaveValue("");
      expect(sourceInput).toHaveAttribute(
        "placeholder",
        "AUTOMATION_SETUP$EVENT_SOURCE_CUSTOM_PLACEHOLDER",
      );
      expect(sourceInput).toHaveFocus();
      fireEvent.change(sourceInput, {
        target: { value: "incident-alerts" },
      });
      expect(
        screen.getByTestId("automation-setup-custom-webhook-enabled"),
      ).toBeChecked();
      await user.click(
        screen.getByTestId("automation-setup-event-source-toggle"),
      );
      expect(
        screen.getByTestId("automation-setup-event-source-option-github"),
      ).toBeInTheDocument();
      await user.click(
        screen.getByTestId("automation-setup-event-source-toggle"),
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
      await saveDraftAndWaitForTest(user);
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

    it("offers each Data Center source's own event types and saves the picked one", async () => {
      vi.mocked(useDeploymentCapabilities).mockReturnValue({
        data: {
          ready: true,
          features: ["automationDrafts", "webhookDelivery"],
          triggerKinds: ["cron", "event"],
          eventSources: ["github", "jira_dc", "bitbucket_data_center"],
          eventTypes: ["issues.*", "issue_comment.*"],
          eventTypesBySource: {
            github: ["issues.*", "issue_comment.*"],
            jira_dc: ["comment_created", "jira:issue_created"],
            bitbucket_data_center: ["pr:comment:added", "pr:opened"],
          },
          triggers: {},
        },
        isLoading: false,
      } as never);
      mockSavedDraftEcho();
      const eventTypeOptions = () =>
        Array.from(
          document.querySelectorAll(
            "#automation-setup-event-key-options option",
          ),
        ).map((option) => option.getAttribute("value"));
      const pickSource = async (
        user: ReturnType<typeof userEvent.setup>,
        source: string,
      ) => {
        await user.click(
          screen.getByTestId("automation-setup-event-source-toggle"),
        );
        await user.click(
          screen.getByTestId(`automation-setup-event-source-option-${source}`),
        );
      };

      const user = userEvent.setup();
      renderPanel();
      await user.click(screen.getByText("AUTOMATIONS$DETAIL$TRIGGER_EVENT"));
      expect(eventTypeOptions()).toEqual(["issues.*", "issue_comment.*"]);
      // GitHub's default key still matches a GitHub pattern.
      expect(screen.getByTestId("automation-setup-event-key")).toHaveValue(
        "issue_comment.created",
      );

      await pickSource(user, "jira_dc");
      expect(eventTypeOptions()).toEqual([
        "comment_created",
        "jira:issue_created",
      ]);
      expect(screen.getByTestId("automation-setup-event-key")).toHaveValue(
        "comment_created",
      );
      await user.click(
        screen.getByTestId("automation-setup-event-test-payload-toggle"),
      );
      expect(
        JSON.parse(
          (
            screen.getByTestId(
              "automation-setup-event-test-payload",
            ) as HTMLTextAreaElement
          ).value,
        ),
      ).toMatchObject({ webhookEvent: "comment_created" });

      await pickSource(user, "bitbucket_data_center");
      expect(eventTypeOptions()).toEqual(["pr:comment:added", "pr:opened"]);
      fireEvent.change(screen.getByTestId("automation-setup-event-key"), {
        target: { value: "pr:opened" },
      });
      expect(
        JSON.parse(
          (
            screen.getByTestId(
              "automation-setup-event-test-payload",
            ) as HTMLTextAreaElement
          ).value,
        ),
      ).toMatchObject({ eventKey: "pr:opened" });
      await saveDraftAndWaitForTest(user);

      expect(AutomationService.createServerDraft).toHaveBeenCalledWith(
        expect.objectContaining({
          draft: expect.objectContaining({
            trigger: expect.objectContaining({
              type: "event",
              source: "bitbucket_data_center",
              on: "pr:opened",
            }),
          }),
        }),
      );
    });

    it("shows the API's reason and drops Ready to test when a draft save is refused", async () => {
      const user = userEvent.setup();
      renderPanel();
      mockSavedDraftEcho();
      await saveDraftAndWaitForTest(user);
      expect(
        screen.getByTestId("automation-setup-draft-validity"),
      ).toHaveTextContent("AUTOMATION_SETUP$READY_TO_TEST");

      vi.mocked(AutomationService.updateServerDraft).mockRejectedValue(
        new HttpError(422, "Unprocessable Entity", {
          detail: {
            message: "Draft body does not match endpoint schema",
            errors: [
              {
                field: "timeout",
                code: "value_error",
                message:
                  "Value error, timeout must not exceed 1800 seconds (30 minutes)",
              },
            ],
          },
        }),
      );
      await user.click(screen.getByTestId("automation-setup-save-draft"));

      await waitFor(() =>
        expect(displayErrorToast).toHaveBeenCalledWith(
          "Value error, timeout must not exceed 1800 seconds (30 minutes)",
        ),
      );
      expect(
        screen.queryByTestId("automation-setup-draft-validity"),
      ).not.toBeInTheDocument();
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
      mockSavedDraftEcho();
      await saveDraftAndWaitForTest(user);
      await user.click(screen.getByTestId("automation-setup-test"));

      expect(displayErrorToast).toHaveBeenCalledWith(
        "AUTOMATION_SETUP$TEST_EVENT_PAYLOAD_INVALID",
      );
      expect(
        screen.queryByTestId("automation-setup-status"),
      ).not.toBeInTheDocument();
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

  it("saves and tests an existing automation instead of drafting or creating one", async () => {
    vi.mocked(AutomationService.updateAutomation).mockResolvedValue({
      id: "auto-1",
    } as never);
    vi.mocked(AutomationService.dispatchAutomation).mockResolvedValue({
      id: "run-9",
    } as never);

    const user = userEvent.setup();
    renderPanel({
      prompt: "Review every pull request",
      kind: "prompt",
      editingAutomationId: "auto-1",
      form: {
        kind: "prompt",
        name: "Daily digest",
        prompt: "Review every pull request",
      },
    });

    expect(
      screen.queryByTestId("automation-setup-save-draft"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-setup-create"),
    ).not.toBeInTheDocument();
    expect(
      [
        ...(screen
          .getByTestId("automation-setup-test")
          .parentElement?.querySelectorAll("button") ?? []),
      ].map((button) => button.getAttribute("data-testid")),
    ).toEqual([
      "automation-setup-test",
      "automation-setup-save",
      "automation-setup-close",
    ]);
    expect(screen.getByTestId("automation-setup-save")).toHaveTextContent(
      "BUTTON$SAVE",
    );
    expect(screen.getByTestId("automation-setup-close")).toHaveTextContent(
      "BUTTON$CLOSE",
    );
    expect(screen.getByTestId("automation-setup-name")).toHaveValue(
      "Daily digest",
    );

    await user.click(screen.getByTestId("automation-setup-save"));
    await waitFor(() =>
      expect(AutomationService.updateAutomation).toHaveBeenCalledWith(
        "auto-1",
        expect.objectContaining({
          name: "Daily digest",
          prompt: "Review every pull request",
        }),
      ),
    );
    expect(AutomationService.updateAutomation).toHaveBeenCalledTimes(1);
    expect(
      vi.mocked(AutomationService.updateAutomation).mock.calls[0][1],
    ).not.toEqual(
      expect.objectContaining({
        repository: expect.anything(),
        repos: expect.anything(),
        plugins: expect.anything(),
      }),
    );
    expect(AutomationService.createServerDraft).not.toHaveBeenCalled();
    expect(AutomationService.createAutomationDraft).not.toHaveBeenCalled();

    await user.click(screen.getByTestId("automation-setup-test"));
    await waitFor(() =>
      expect(AutomationService.dispatchAutomation).toHaveBeenCalledWith(
        "auto-1",
      ),
    );
    expect(AutomationService.updateAutomation).toHaveBeenCalledTimes(1);

    await user.click(screen.getByTestId("automation-setup-close"));
    expect(mockNavigate).toHaveBeenCalledWith("/automations");
  });

  describe("existing automation request failures", () => {
    const editDraft: AutomationSetupDraft = {
      prompt: "Review every pull request",
      kind: "prompt",
      editingAutomationId: "auto-1",
      form: {
        kind: "prompt",
        name: "Daily digest",
        prompt: "Review every pull request",
      },
    };

    it.each([
      {
        action: "Save",
        button: "automation-setup-save",
        service: "updateAutomation",
        error: axiosResponseError(422, {
          detail: [
            {
              loc: ["body", "trigger", "schedule"],
              msg: "Value error, Invalid cron expression: not a cron",
              type: "value_error",
            },
          ],
        }),
        expected: "Value error, Invalid cron expression: not a cron",
      },
      {
        action: "Test",
        button: "automation-setup-test",
        service: "dispatchAutomation",
        error: axiosResponseError(404, { detail: "Automation not found" }),
        expected: "Automation not found",
      },
      {
        action: "Save (Cloud)",
        button: "automation-setup-save",
        service: "updateAutomation",
        error: new HttpError(
          404,
          "Not Found",
          { detail: "Automation not found" },
          `HTTP request failed (404 Not Found): {"detail":"Automation not found"}`,
        ),
        expected: "Automation not found",
      },
    ] as const)(
      "toasts the API's message when $action fails",
      async ({ button, service, error, expected }) => {
        vi.mocked(AutomationService[service]).mockRejectedValue(error);
        const user = userEvent.setup();
        renderPanel(editDraft);

        await user.click(screen.getByTestId(button));

        await waitFor(() =>
          expect(displayErrorToast).toHaveBeenCalledWith(expected),
        );
        expect(displayErrorToast).toHaveBeenCalledTimes(1);
      },
    );
  });

  it("toasts the API's message when creating an automation fails", async () => {
    vi.mocked(AutomationService.createAutomationDraft).mockRejectedValue(
      axiosResponseError(422, {
        detail: [
          {
            loc: ["body", "timeout"],
            msg: "Input should be less than or equal to 1800",
            type: "less_than_equal",
          },
        ],
      }),
    );
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByTestId("automation-setup-create"));

    await waitFor(() =>
      expect(displayErrorToast).toHaveBeenCalledWith(
        "Input should be less than or equal to 1800",
      ),
    );
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it("preserves multi-event trigger keys when saving an existing automation", async () => {
    vi.mocked(AutomationService.updateAutomation).mockResolvedValue({
      id: "auto-1",
    } as never);

    const user = userEvent.setup();
    renderPanel(
      setupDraftFromAutomation({
        id: "auto-1",
        name: "Triage",
        prompt: "Triage it",
        enabled: true,
        created_at: "2026-01-01T00:00:00.000Z",
        updated_at: "2026-01-01T00:00:00.000Z",
        trigger: {
          type: "event",
          source: "github",
          on: ["pull_request", "issues"],
        },
      }),
    );

    await user.click(screen.getByTestId("automation-setup-save"));

    await waitFor(() =>
      expect(AutomationService.updateAutomation).toHaveBeenCalledWith(
        "auto-1",
        expect.objectContaining({
          trigger: expect.objectContaining({
            on: ["pull_request", "issues"],
          }),
        }),
      ),
    );
  });

  it("explains existing event automation tests use the backend payload", async () => {
    vi.mocked(AutomationService.updateAutomation).mockResolvedValue({
      id: "auto-1",
    } as never);
    vi.mocked(AutomationService.dispatchAutomation).mockResolvedValue({
      id: "run-1",
    } as never);

    const user = userEvent.setup();
    renderPanel(
      setupDraftFromAutomation({
        id: "auto-1",
        name: "Triage",
        prompt: "Triage it",
        enabled: true,
        created_at: "2026-01-01T00:00:00.000Z",
        updated_at: "2026-01-01T00:00:00.000Z",
        trigger: {
          type: "event",
          source: "github",
          on: "issues",
        },
      }),
    );

    expect(
      screen.getByTestId("automation-setup-existing-test-payload-note"),
    ).toHaveTextContent("AUTOMATION_SETUP$EXISTING_TEST_PAYLOAD_NOTICE");
    expect(
      screen.queryByTestId("automation-setup-event-test-payload-toggle"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-setup-event-test-payload"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByTestId("automation-setup-test"));

    await waitFor(() =>
      expect(AutomationService.dispatchAutomation).toHaveBeenCalledWith(
        "auto-1",
      ),
    );
    expect(AutomationService.dispatchServerDraft).not.toHaveBeenCalled();
  });

  const savedReviewer = (overrides: Partial<Automation> = {}): Automation => ({
    id: "auto-1",
    name: "Reviewer",
    prompt: "Review PRs",
    enabled: true,
    created_at: "2026-01-01T00:00:00.000Z",
    updated_at: "2026-01-01T00:00:00.000Z",
    // The service keeps repositories in preset metadata; the client reads
    // them as `repositories`.
    repositories: [
      { url: "OpenHands/OpenHands", ref: "main", provider: "github" },
    ],
    preset_metadata: {
      preset_type: "prompt",
      repos: [{ url: "OpenHands/OpenHands", ref: "main", provider: "github" }],
    },
    trigger: {
      type: "cron",
      schedule: "0 9 * * *",
      timezone: "UTC",
    },
    ...overrides,
  });

  it("shows saved repositories read-only when the service cannot change them", async () => {
    vi.mocked(AutomationService.updateAutomation).mockResolvedValue(
      savedReviewer(),
    );

    const user = userEvent.setup();
    renderPanel(
      setupDraftFromAutomation(
        savedReviewer({
          plugins: ["github:OpenHands/extensions/plugins/pr-review"],
        }),
      ),
    );

    expect(
      screen.getByTestId("automation-setup-repository-value"),
    ).toHaveTextContent("OpenHands/OpenHands");
    expect(
      screen.getByTestId("automation-setup-repository-read-only"),
    ).toHaveTextContent("AUTOMATION_SETUP$REPOSITORIES_READ_ONLY");
    expect(
      screen.queryByTestId("automation-setup-repository-add"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-setup-repository-remove"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByTestId("automation-setup-plugin-remove"));
    await user.click(screen.getByTestId("automation-setup-save"));

    await waitFor(() =>
      expect(AutomationService.updateAutomation).toHaveBeenCalledWith(
        "auto-1",
        expect.objectContaining({
          name: "Reviewer",
          prompt: "Review PRs",
        }),
      ),
    );
    expect(
      vi.mocked(AutomationService.updateAutomation).mock.calls[0][1],
    ).not.toEqual(
      expect.objectContaining({
        repository: expect.anything(),
        repos: expect.anything(),
        plugins: expect.anything(),
      }),
    );
  });

  it("saves changed repositories when the service can change them", async () => {
    vi.mocked(useDeploymentCapabilities).mockReturnValue({
      data: {
        ready: true,
        features: ["repoUpdate"],
        triggerKinds: ["cron"],
        eventSources: [],
        eventTypes: [],
        triggers: {},
      },
      isLoading: false,
    } as never);
    vi.mocked(AutomationService.updateAutomation).mockImplementation(
      async (_id, body) =>
        savedReviewer({
          preset_metadata: {
            preset_type: "prompt",
            ...((body as { repos?: unknown }).repos
              ? { repos: (body as { repos: unknown }).repos }
              : {}),
          },
        }),
    );

    const user = userEvent.setup();
    renderPanel(setupDraftFromAutomation(savedReviewer()));

    // An unchanged list is not sent.
    await user.click(screen.getByTestId("automation-setup-save"));
    await waitFor(() =>
      expect(AutomationService.updateAutomation).toHaveBeenCalledTimes(1),
    );
    expect(
      vi.mocked(AutomationService.updateAutomation).mock.calls[0][1],
    ).not.toHaveProperty("repos");

    await user.click(screen.getByTestId("automation-setup-repository-add"));
    await user.click(
      await screen.findByTestId(
        "automation-setup-repository-option-OpenHands/software-agent-sdk",
      ),
    );
    await user.click(screen.getByTestId("automation-setup-save"));

    // The saved repository keeps its branch and provider.
    await waitFor(() =>
      expect(AutomationService.updateAutomation).toHaveBeenCalledTimes(2),
    );
    expect(
      vi.mocked(AutomationService.updateAutomation).mock.calls[1][1],
    ).toMatchObject({
      repos: [
        { url: "OpenHands/OpenHands", ref: "main", provider: "github" },
        { url: "OpenHands/software-agent-sdk", provider: "github" },
      ],
    });
  });

  it("saves a repository removal made after an earlier save", async () => {
    vi.mocked(useDeploymentCapabilities).mockReturnValue({
      data: {
        ready: true,
        features: ["repoUpdate"],
        triggerKinds: ["cron"],
        eventSources: [],
        eventTypes: [],
        triggers: {},
      },
      isLoading: false,
    } as never);
    vi.mocked(AutomationService.updateAutomation).mockImplementation(
      async (_id, body) =>
        savedReviewer({
          preset_metadata: {
            preset_type: "prompt",
            repos: (body as { repos: unknown }).repos,
          },
        }),
    );
    initializeAutomationFormSession(
      PENDING_AUTOMATION_SETUP_ID,
      setupDraftFromAutomation(savedReviewer()),
    );

    const user = userEvent.setup();
    renderSessionBoundPanel(PENDING_AUTOMATION_SETUP_ID);

    await user.click(screen.getByTestId("automation-setup-repository-add"));
    await user.click(
      await screen.findByTestId(
        "automation-setup-repository-option-OpenHands/software-agent-sdk",
      ),
    );
    await user.click(screen.getByTestId("automation-setup-save"));
    await waitFor(() =>
      expect(AutomationService.updateAutomation).toHaveBeenCalledTimes(1),
    );

    await user.click(
      screen.getByRole("button", {
        name: "COMMON$REMOVE OpenHands/software-agent-sdk",
      }),
    );
    await user.click(screen.getByTestId("automation-setup-save"));

    await waitFor(() =>
      expect(AutomationService.updateAutomation).toHaveBeenCalledTimes(2),
    );
    expect(
      vi.mocked(AutomationService.updateAutomation).mock.calls[1][1],
    ).toMatchObject({
      repos: [{ url: "OpenHands/OpenHands", ref: "main", provider: "github" }],
    });
    clearAutomationFormSession(PENDING_AUTOMATION_SETUP_ID);
  });

  it("refreshes the cached automation after a save, so Edit from its page starts from the saved repositories", async () => {
    vi.mocked(useDeploymentCapabilities).mockReturnValue({
      data: {
        ready: true,
        features: ["repoUpdate"],
        triggerKinds: ["cron"],
        eventSources: [],
        eventTypes: [],
        triggers: {},
      },
      isLoading: false,
    } as never);
    vi.mocked(AutomationService.updateAutomation).mockImplementation(
      async (_id, body) =>
        savedReviewer({
          preset_metadata: {
            preset_type: "prompt",
            repos: (body as { repos: unknown }).repos,
          },
        }),
    );
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    // The automation's page keeps its copy for five minutes and opens Edit from it.
    const detailKey = [
      ...AUTOMATION_DETAIL_QUERY_KEY,
      "auto-1",
      "cloud",
      "org-1",
    ];
    const listKey = [...AUTOMATIONS_QUERY_KEY, "cloud", "org-1"];
    queryClient.setQueryData(detailKey, savedReviewer());
    queryClient.setQueryData(listKey, { automations: [savedReviewer()] });

    const user = userEvent.setup();
    renderPanel(
      setupDraftFromAutomation(savedReviewer()),
      "conv-1",
      null,
      queryClient,
    );
    await user.click(screen.getByTestId("automation-setup-repository-add"));
    await user.click(
      await screen.findByTestId(
        "automation-setup-repository-option-OpenHands/software-agent-sdk",
      ),
    );
    await user.click(screen.getByTestId("automation-setup-save"));
    await waitFor(() =>
      expect(AutomationService.updateAutomation).toHaveBeenCalledTimes(1),
    );

    await waitFor(() =>
      expect(
        setupDraftFromAutomation(
          queryClient.getQueryData<Automation>(detailKey)!,
        ).form?.repository,
      ).toBe("OpenHands/OpenHands, OpenHands/software-agent-sdk"),
    );
    expect(queryClient.getQueryState(detailKey)?.isInvalidated).toBe(true);
    expect(queryClient.getQueryState(listKey)?.isInvalidated).toBe(true);
  });

  it("shows existing custom bundle metadata and saves a model change without replacing the bundle", async () => {
    vi.mocked(AutomationService.updateAutomation).mockResolvedValue({
      id: "auto-1",
    } as never);

    const user = userEvent.setup();
    renderPanel(
      setupDraftFromAutomation({
        id: "auto-1",
        name: "Nightly scan",
        prompt: null,
        enabled: true,
        created_at: "2026-01-01T00:00:00.000Z",
        updated_at: "2026-01-01T00:00:00.000Z",
        entrypoint: "python3 scan.py",
        setup_script_path: "install.sh",
        tarball_path: "oh-internal://uploads/REAL-user-code.tar.gz",
        model: "fast",
        trigger: {
          type: "cron",
          schedule: "0 9 * * *",
          timezone: "UTC",
        },
      }),
    );

    expect(
      screen.getByTestId("automation-setup-existing-bundle"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("automation-setup-existing-bundle-name"),
    ).toHaveTextContent("REAL-user-code.tar.gz");
    expect(
      screen.getByTestId("automation-setup-existing-bundle-info"),
    ).toHaveAccessibleName("AUTOMATION_SETUP$EXISTING_BUNDLE_HELP");
    expect(
      screen.queryByTestId("automation-setup-custom-code"),
    ).not.toBeInTheDocument();

    expect(screen.getByTestId("automation-setup-model")).toHaveAttribute(
      "aria-label",
      "fast",
    );
    await user.click(screen.getByTestId("automation-setup-model"));
    await user.click(
      screen.getByTestId(
        "automation-setup-model-option-claude-opus-4-5-20251101",
      ),
    );
    await user.click(screen.getByTestId("automation-setup-save"));

    await waitFor(() =>
      expect(AutomationService.updateAutomation).toHaveBeenCalledWith(
        "auto-1",
        expect.objectContaining({
          name: "Nightly scan",
          model: "claude-opus-4-5-20251101",
        }),
      ),
    );
    expect(
      vi.mocked(AutomationService.updateAutomation).mock.calls[0][1],
    ).not.toEqual(expect.objectContaining({ entrypoint: expect.anything() }));
    expect(AutomationService.uploadAutomationTarball).not.toHaveBeenCalled();
    expect(packTarGzip).not.toHaveBeenCalled();
    expect(
      vi.mocked(AutomationService.updateAutomation).mock.calls[0][1],
    ).not.toEqual(expect.objectContaining({ tarball_path: expect.anything() }));
    expect(
      vi.mocked(AutomationService.updateAutomation).mock.calls[0][1],
    ).not.toEqual(
      expect.objectContaining({ setup_script_path: expect.anything() }),
    );
  });
});
