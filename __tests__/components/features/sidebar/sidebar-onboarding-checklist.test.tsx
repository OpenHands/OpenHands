import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  type MockInstance,
  vi,
} from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ONBOARDING_COMPLETED_STORAGE_KEY } from "#/components/features/onboarding/use-onboarding-completion";
import { SidebarOnboardingChecklist } from "#/components/features/sidebar/sidebar-onboarding-checklist";
import {
  OPENHANDS_SLACK_COMMUNITY_URL,
  SIDEBAR_ONBOARDING_CHECKLIST_CUSTOMIZE_EXPLORED_STORAGE_KEY,
  SIDEBAR_ONBOARDING_CHECKLIST_MINIMIZED_STORAGE_KEY,
  SIDEBAR_ONBOARDING_CHECKLIST_SLACK_JOINED_STORAGE_KEY,
} from "#/components/features/sidebar/sidebar-onboarding-checklist.constants";
import {
  readSidebarOnboardingChecklistMinimized,
  readSidebarOnboardingChecklistSlackJoined,
} from "#/components/features/sidebar/sidebar-onboarding-checklist-storage";
import {
  NavigationProvider,
  type NavigationContextValue,
} from "#/context/navigation-context";
import { I18nKey } from "#/i18n/declaration";
import * as telemetry from "#/services/telemetry";

const mockUsePaginatedConversations = vi.fn();
const mockUseAutomationHealth = vi.fn();
const mockUseAutomations = vi.fn();
const mockUseSettings = vi.fn();
const mockUseLlmConfigured = vi.fn();
const mockUseLlmProfiles = vi.fn();

vi.mock("#/hooks/query/use-paginated-conversations", () => ({
  usePaginatedConversations: () => mockUsePaginatedConversations(),
}));

vi.mock("#/hooks/query/use-automation-health", () => ({
  useAutomationHealth: () => mockUseAutomationHealth(),
}));

vi.mock("#/hooks/query/use-automations", () => ({
  useAutomations: () => mockUseAutomations(),
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => mockUseSettings(),
}));

vi.mock("#/hooks/use-llm-configured", () => ({
  useLlmConfigured: () => mockUseLlmConfigured(),
}));

vi.mock("#/hooks/query/use-llm-profiles", () => ({
  useLlmProfiles: () => mockUseLlmProfiles(),
}));

function renderChecklist() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const navigate = vi.fn();
  const navigation: NavigationContextValue = {
    currentPath: "/",
    conversationId: null,
    isNavigating: false,
    navigate,
  };

  return {
    ...render(
      <QueryClientProvider client={queryClient}>
        <NavigationProvider value={navigation}>
          <SidebarOnboardingChecklist collapsed={false} />
        </NavigationProvider>
      </QueryClientProvider>,
    ),
    navigate,
  };
}

describe("SidebarOnboardingChecklist", () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.localStorage.setItem(ONBOARDING_COMPLETED_STORAGE_KEY, "1");
    window.localStorage.removeItem(SIDEBAR_ONBOARDING_CHECKLIST_MINIMIZED_STORAGE_KEY);
    window.localStorage.removeItem(
      SIDEBAR_ONBOARDING_CHECKLIST_CUSTOMIZE_EXPLORED_STORAGE_KEY,
    );
    window.localStorage.removeItem(
      SIDEBAR_ONBOARDING_CHECKLIST_SLACK_JOINED_STORAGE_KEY,
    );

    mockUsePaginatedConversations.mockReturnValue({
      data: { pages: [{ items: [{ id: "conv-1" }] }] },
    });
    mockUseAutomationHealth.mockReturnValue({
      data: { status: "ok" },
    });
    mockUseAutomations.mockReturnValue({
      data: { total: 0, automations: [] },
    });
    mockUseSettings.mockReturnValue({
      data: {
        agent_settings: {
          mcp_config: { mcpServers: {} },
        },
      },
    });
    mockUseLlmConfigured.mockReturnValue({
      isConfigured: false,
      isLoading: false,
    });
    mockUseLlmProfiles.mockReturnValue({
      data: { active_profile: null, profiles: [] },
      isLoading: false,
    });
  });

  it("renders setup items including LLM keys, agent profiles, schedule a task, and Slack", () => {
    renderChecklist();

    expect(
      screen.getByTestId("sidebar-onboarding-checklist"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("sidebar-onboarding-checklist-item-configure-llm"),
    ).toHaveAttribute("href", "/settings/llm");
    expect(
      screen.getByTestId("sidebar-onboarding-checklist-item-connect-mcp"),
    ).toHaveAttribute("href", "/mcp");
    expect(
      screen.getByTestId("sidebar-onboarding-checklist-item-schedule-task"),
    ).toHaveAttribute("href", "/automations");
    expect(
      screen.getByTestId("sidebar-onboarding-checklist-item-customize-agent"),
    ).toHaveAttribute("href", "/settings/agents");
    expect(
      screen.getByTestId("sidebar-onboarding-checklist-item-join-slack"),
    ).toHaveAttribute("href", OPENHANDS_SLACK_COMMUNITY_URL);
    expect(
      screen.getByTestId("sidebar-onboarding-checklist-item-join-slack"),
    ).toHaveAttribute("target", "_blank");
    expect(
      screen.getByText(I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_CONFIGURE_LLM),
    ).toBeInTheDocument();
    expect(
      screen.getByText(I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_JOIN_SLACK),
    ).toBeInTheDocument();
  });

  it("marks Join Slack complete after the invite link is clicked", async () => {
    const user = userEvent.setup();
    renderChecklist();

    const slackItem = screen.getByTestId(
      "sidebar-onboarding-checklist-item-join-slack",
    );
    expect(
      screen.getByText(I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_JOIN_SLACK),
    ).not.toHaveClass("line-through");

    await user.click(slackItem);

    expect(readSidebarOnboardingChecklistSlackJoined()).toBe(true);
    expect(
      screen.getByText(I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_JOIN_SLACK),
    ).toHaveClass("line-through");
  });

  it("crosses out Add LLM API key when LLM is configured", () => {
    mockUseLlmConfigured.mockReturnValue({
      isConfigured: true,
      isLoading: false,
    });
    mockUseLlmProfiles.mockReturnValue({
      data: {
        active_profile: "work",
        profiles: [
          {
            name: "work",
            model: "openai/gpt-5.5",
            base_url: "https://api.openai.com/v1",
            api_key_set: true,
          },
        ],
      },
      isLoading: false,
    });
    mockUseSettings.mockReturnValue({
      data: {
        llm_api_key_set: true,
        agent_settings: {
          llm: { model: "openai/gpt-5.5" },
          mcp_config: { mcpServers: {} },
        },
      },
    });

    renderChecklist();

    expect(
      screen.getByText(I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_CONFIGURE_LLM),
    ).toHaveClass("line-through");
  });

  it("crosses out Add LLM API key when a saved profile has an API key", () => {
    mockUseLlmConfigured.mockReturnValue({
      isConfigured: false,
      isLoading: true,
    });
    mockUseLlmProfiles.mockReturnValue({
      data: {
        active_profile: "work",
        profiles: [
          {
            name: "work",
            model: "openai/gpt-5.5",
            base_url: "https://api.openai.com/v1",
            api_key_set: true,
          },
        ],
      },
      isLoading: false,
    });
    mockUseSettings.mockReturnValue({
      data: {
        agent_settings: {
          mcp_config: { mcpServers: {} },
        },
      },
    });

    renderChecklist();

    expect(
      screen.getByText(I18nKey.SIDEBAR$ONBOARDING_CHECKLIST_CONFIGURE_LLM),
    ).toHaveClass("line-through");
  });

  it("hides when the welcome onboarding flow is not complete", () => {
    window.localStorage.removeItem(ONBOARDING_COMPLETED_STORAGE_KEY);
    renderChecklist();

    expect(
      screen.queryByTestId("sidebar-onboarding-checklist"),
    ).not.toBeInTheDocument();
  });

  it("minimizes and expands with the caret toggle", async () => {
    const user = userEvent.setup();
    renderChecklist();

    expect(
      screen.getByTestId("sidebar-onboarding-checklist-item-schedule-task"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("sidebar-onboarding-checklist-toggle"));

    expect(
      screen.queryByTestId("sidebar-onboarding-checklist-item-schedule-task"),
    ).not.toBeInTheDocument();
    expect(readSidebarOnboardingChecklistMinimized()).toBe(true);

    await user.click(screen.getByTestId("sidebar-onboarding-checklist-toggle"));

    expect(
      screen.getByTestId("sidebar-onboarding-checklist-item-schedule-task"),
    ).toBeInTheDocument();
    expect(readSidebarOnboardingChecklistMinimized()).toBe(false);
  });

  it("hides when collapsed", () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    const navigation: NavigationContextValue = {
      currentPath: "/",
      conversationId: null,
      isNavigating: false,
      navigate: vi.fn(),
    };

    render(
      <QueryClientProvider client={queryClient}>
        <NavigationProvider value={navigation}>
          <SidebarOnboardingChecklist collapsed />
        </NavigationProvider>
      </QueryClientProvider>,
    );

    expect(
      screen.queryByTestId("sidebar-onboarding-checklist"),
    ).not.toBeInTheDocument();
  });

  describe("onboarding link tracking", () => {
    let captureMock: MockInstance<typeof telemetry.trackEvent>;

    beforeEach(() => {
      captureMock = vi
        .spyOn(telemetry, "trackEvent")
        .mockResolvedValue(undefined);
    });

    afterEach(() => {
      captureMock.mockRestore();
    });

    const linkClickEvents = () =>
      captureMock.mock.calls.filter(
        ([name]) => name === "onboarding_link_clicked",
      );

    it("captures a single onboarding_link_clicked when the Join Slack row is clicked", async () => {
      const user = userEvent.setup();
      renderChecklist();

      await user.click(
        screen.getByTestId("sidebar-onboarding-checklist-item-join-slack"),
      );

      expect(linkClickEvents()).toHaveLength(1);
      expect(linkClickEvents()[0][1]).toMatchObject({
        link_id: "join_slack",
        destination_type: "community",
        surface: "landing_checklist",
        checklist_item: "join_slack",
        is_external: true,
      });
    });

    it("captures onboarding_link_clicked for an internal row and still navigates", async () => {
      const user = userEvent.setup();
      const { navigate } = renderChecklist();

      await user.click(
        screen.getByTestId("sidebar-onboarding-checklist-item-configure-llm"),
      );

      expect(linkClickEvents()).toHaveLength(1);
      expect(linkClickEvents()[0][1]).toMatchObject({
        link_id: "configure_llm",
        destination_type: "settings",
        surface: "landing_checklist",
        checklist_item: "configure_llm",
        is_external: false,
      });
      expect(navigate).toHaveBeenCalledWith("/settings/llm", {
        replace: false,
      });
    });

    it("captures open_docs with the owning checklist item when a preview docs link is clicked", async () => {
      const user = userEvent.setup();
      renderChecklist();

      await user.hover(
        screen.getByTestId("sidebar-onboarding-checklist-item-connect-mcp"),
      );
      await user.click(
        await screen.findByTestId(
          "sidebar-onboarding-checklist-preview-docs-connect-mcp",
        ),
      );

      expect(linkClickEvents()).toHaveLength(1);
      expect(linkClickEvents()[0][1]).toMatchObject({
        link_id: "open_docs",
        destination_type: "documentation",
        surface: "landing_checklist",
        checklist_item: "connect_mcp",
        is_external: true,
      });
    });

    it("captures the row's link_id when a preview action CTA is clicked", async () => {
      const user = userEvent.setup();
      renderChecklist();

      await user.hover(
        screen.getByTestId("sidebar-onboarding-checklist-item-schedule-task"),
      );
      await user.click(
        await screen.findByTestId(
          "sidebar-onboarding-checklist-preview-action-schedule-task",
        ),
      );

      expect(linkClickEvents()).toHaveLength(1);
      expect(linkClickEvents()[0][1]).toMatchObject({
        link_id: "schedule_task",
        destination_type: "automation",
        surface: "landing_checklist",
        checklist_item: "schedule_task",
        is_external: false,
      });
    });
  });
});
