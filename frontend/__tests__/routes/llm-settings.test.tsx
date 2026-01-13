import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClientProvider, QueryClient } from "@tanstack/react-query";
import LlmSettingsScreen from "#/routes/llm-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  MOCK_DEFAULT_USER_SETTINGS,
  resetTestHandlersMockSettings,
} from "#/mocks/handlers";
import * as AdvancedSettingsUtlls from "#/utils/has-advanced-settings-set";
import * as ToastHandlers from "#/utils/custom-toast-handlers";
import OptionService from "#/api/option-service/option-service.api";
import { organizationService } from "#/api/organization-service/organization-service.api";
import { useSelectedOrganizationStore } from "#/stores/selected-organization-store";
import type { OrganizationUserRole, OrganizationMember } from "#/types/org";

// Mock react-router hooks
const mockUseSearchParams = vi.fn();
vi.mock("react-router", async () => {
  const actual =
    await vi.importActual<typeof import("react-router")>("react-router");
  return {
    ...actual,
    useSearchParams: () => mockUseSearchParams(),
    useRevalidator: () => ({
      revalidate: vi.fn(),
    }),
  };
});

// Mock useIsAuthed hook
const mockUseIsAuthed = vi.fn();
vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => mockUseIsAuthed(),
}));

// Mock useConfig hook
const mockUseConfig = vi.fn();
vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => mockUseConfig(),
}));

const renderLlmSettingsScreen = (
  orgId: string | null = null,
  meData?: {
    org_id: string;
    user_id: string;
    email: string;
    role: string;
    status: string;
    llm_api_key: string;
    max_iterations: number;
    llm_model: string;
    llm_api_key_for_byor: string | null;
    llm_base_url: string;
  },
) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
  // Default to orgId "1" if not provided (for backward compatibility)
  const finalOrgId = orgId ?? "1";
  useSelectedOrganizationStore.setState({ organizationId: finalOrgId });

  // Pre-populate React Query cache with me data
  // If meData is provided, use it; otherwise use default owner data
  const defaultMeData = {
    org_id: finalOrgId,
    user_id: "99",
    email: "owner@example.com",
    role: "owner",
    status: "active",
    llm_api_key: "",
    max_iterations: 20,
    llm_model: "",
    llm_api_key_for_byor: null,
    llm_base_url: "",
  };
  queryClient.setQueryData(
    ["organizations", finalOrgId, "me"],
    meData || defaultMeData,
  );

  return render(<LlmSettingsScreen />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
};

beforeEach(() => {
  vi.resetAllMocks();
  resetTestHandlersMockSettings();

  // Default mock for useSearchParams - returns empty params
  mockUseSearchParams.mockReturnValue([
    {
      get: () => null,
    },
    vi.fn(),
  ]);

  // Default mock for useIsAuthed - returns authenticated by default
  mockUseIsAuthed.mockReturnValue({ data: true, isLoading: false });

  // Default mock for useConfig - returns SaaS mode by default
  mockUseConfig.mockReturnValue({
    data: { APP_MODE: "saas" },
    isLoading: false,
  });

  // Default mock for organizationService.getMe - returns owner role by default (full access)
  const defaultMeData: OrganizationMember = {
    org_id: "1",
    user_id: "99",
    email: "owner@example.com",
    role: "owner",
    status: "active",
    llm_api_key: "",
    max_iterations: 20,
    llm_model: "",
    llm_api_key_for_byor: null,
    llm_base_url: "",
  };
  vi.spyOn(organizationService, "getMe").mockResolvedValue(defaultMeData);

  // Reset organization store
  useSelectedOrganizationStore.setState({ organizationId: "1" });
});

describe("Content", () => {
  describe("Basic form", () => {
    it("should render the basic form by default", async () => {
      // Use OSS mode so API key input is visible
      mockUseConfig.mockReturnValue({
        data: { APP_MODE: "oss" },
        isLoading: false,
      });

      renderLlmSettingsScreen();
      await screen.findByTestId("llm-settings-screen");

      const basicFom = screen.getByTestId("llm-settings-form-basic");
      within(basicFom).getByTestId("llm-provider-input");
      within(basicFom).getByTestId("llm-model-input");
      within(basicFom).getByTestId("llm-api-key-input");
      within(basicFom).getByTestId("llm-api-key-help-anchor");
    });

    it("should render the default values if non exist", async () => {
      // Use OSS mode so API key input is visible
      mockUseConfig.mockReturnValue({
        data: { APP_MODE: "oss" },
        isLoading: false,
      });

      renderLlmSettingsScreen();
      await screen.findByTestId("llm-settings-screen");

      const provider = screen.getByTestId("llm-provider-input");
      const model = screen.getByTestId("llm-model-input");
      const apiKey = screen.getByTestId("llm-api-key-input");

      await waitFor(() => {
        expect(provider).toHaveValue("OpenHands");
        expect(model).toHaveValue("claude-opus-4-5-20251101");

        expect(apiKey).toHaveValue("");
        expect(apiKey).toHaveProperty("placeholder", "");
      });
    });

    it("should render the existing settings values", async () => {
      const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
      getSettingsSpy.mockResolvedValue({
        ...MOCK_DEFAULT_USER_SETTINGS,
        llm_model: "openai/gpt-4o",
        llm_api_key_set: true,
      });

      renderLlmSettingsScreen();
      await screen.findByTestId("llm-settings-screen");

      const provider = screen.getByTestId("llm-provider-input");
      const model = screen.getByTestId("llm-model-input");
      const apiKey = screen.getByTestId("llm-api-key-input");

      await waitFor(() => {
        expect(provider).toHaveValue("OpenAI");
        expect(model).toHaveValue("gpt-4o");

        expect(apiKey).toHaveValue("");
        expect(apiKey).toHaveProperty("placeholder", "<hidden>");
        expect(screen.getByTestId("set-indicator")).toBeInTheDocument();
      });
    });
  });

  describe("Advanced form", () => {
    it("should conditionally show security analyzer based on confirmation mode", async () => {
      renderLlmSettingsScreen();
      await screen.findByTestId("llm-settings-screen");

      // Enable advanced mode first
      const advancedSwitch = screen.getByTestId("advanced-settings-switch");
      await userEvent.click(advancedSwitch);

      const confirmation = screen.getByTestId(
        "enable-confirmation-mode-switch",
      );

      // Initially confirmation mode is false, so security analyzer should not be visible
      expect(confirmation).not.toBeChecked();
      expect(
        screen.queryByTestId("security-analyzer-input"),
      ).not.toBeInTheDocument();

      // Enable confirmation mode
      await userEvent.click(confirmation);
      expect(confirmation).toBeChecked();

      // Security analyzer should now be visible
      screen.getByTestId("security-analyzer-input");

      // Disable confirmation mode again
      await userEvent.click(confirmation);
      expect(confirmation).not.toBeChecked();

      // Security analyzer should be hidden again
      expect(
        screen.queryByTestId("security-analyzer-input"),
      ).not.toBeInTheDocument();
    });

    it("should render the advanced form if the switch is toggled", async () => {
      // Use OSS mode so agent-input is visible
      mockUseConfig.mockReturnValue({
        data: { APP_MODE: "oss" },
        isLoading: false,
      });

      renderLlmSettingsScreen();
      await screen.findByTestId("llm-settings-screen");

      const advancedSwitch = screen.getByTestId("advanced-settings-switch");
      const basicForm = screen.getByTestId("llm-settings-form-basic");

      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
      expect(basicForm).toBeInTheDocument();

      await userEvent.click(advancedSwitch);

      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).toBeInTheDocument();
      expect(basicForm).not.toBeInTheDocument();

      const advancedForm = screen.getByTestId("llm-settings-form-advanced");
      within(advancedForm).getByTestId("llm-custom-model-input");
      within(advancedForm).getByTestId("base-url-input");
      within(advancedForm).getByTestId("llm-api-key-input");
      within(advancedForm).getByTestId("llm-api-key-help-anchor-advanced");
      within(advancedForm).getByTestId("agent-input");
      within(advancedForm).getByTestId("enable-memory-condenser-switch");

      await userEvent.click(advancedSwitch);
      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
      expect(screen.getByTestId("llm-settings-form-basic")).toBeInTheDocument();
    });

    it("should render the default advanced settings", async () => {
      // Use OSS mode so agent-input is visible
      mockUseConfig.mockReturnValue({
        data: { APP_MODE: "oss" },
        isLoading: false,
      });

      renderLlmSettingsScreen();
      await screen.findByTestId("llm-settings-screen");

      const advancedSwitch = screen.getByTestId("advanced-settings-switch");
      expect(advancedSwitch).not.toBeChecked();

      await userEvent.click(advancedSwitch);

      const model = screen.getByTestId("llm-custom-model-input");
      const baseUrl = screen.getByTestId("base-url-input");
      const apiKey = screen.getByTestId("llm-api-key-input");
      const agent = screen.getByTestId("agent-input");
      const condensor = screen.getByTestId("enable-memory-condenser-switch");

      expect(model).toHaveValue("openhands/claude-opus-4-5-20251101");
      expect(baseUrl).toHaveValue("");
      expect(apiKey).toHaveValue("");
      expect(apiKey).toHaveProperty("placeholder", "");
      expect(agent).toHaveValue("CodeActAgent");
      expect(condensor).toBeChecked();
    });

    it("should render the advanced form if existings settings are advanced", async () => {
      const hasAdvancedSettingsSetSpy = vi.spyOn(
        AdvancedSettingsUtlls,
        "hasAdvancedSettingsSet",
      );
      hasAdvancedSettingsSetSpy.mockReturnValue(true);

      renderLlmSettingsScreen();

      await waitFor(() => {
        const advancedSwitch = screen.getByTestId("advanced-settings-switch");
        expect(advancedSwitch).toBeChecked();
        screen.getByTestId("llm-settings-form-advanced");
      });
    });

    it("should render existing advanced settings correctly", async () => {
      // Use OSS mode so agent-input is visible
      mockUseConfig.mockReturnValue({
        data: { APP_MODE: "oss" },
        isLoading: false,
      });

      const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
      getSettingsSpy.mockResolvedValue({
        ...MOCK_DEFAULT_USER_SETTINGS,
        llm_model: "openai/gpt-4o",
        llm_base_url: "https://api.openai.com/v1/chat/completions",
        llm_api_key_set: true,
        agent: "CoActAgent",
        confirmation_mode: true,
        enable_default_condenser: false,
        security_analyzer: "none",
      });

      renderLlmSettingsScreen();
      await screen.findByTestId("llm-settings-screen");

      const model = screen.getByTestId("llm-custom-model-input");
      const baseUrl = screen.getByTestId("base-url-input");
      const apiKey = screen.getByTestId("llm-api-key-input");
      const agent = screen.getByTestId("agent-input");
      const confirmation = screen.getByTestId(
        "enable-confirmation-mode-switch",
      );
      const condensor = screen.getByTestId("enable-memory-condenser-switch");
      const securityAnalyzer = screen.getByTestId("security-analyzer-input");

      await waitFor(() => {
        expect(model).toHaveValue("openai/gpt-4o");
        expect(baseUrl).toHaveValue(
          "https://api.openai.com/v1/chat/completions",
        );
        expect(apiKey).toHaveValue("");
        expect(apiKey).toHaveProperty("placeholder", "<hidden>");
        expect(agent).toHaveValue("CoActAgent");
        expect(confirmation).toBeChecked();
        expect(condensor).not.toBeChecked();
        expect(securityAnalyzer).toHaveValue("SETTINGS$SECURITY_ANALYZER_NONE");
      });
    });

    it("should omit invariant and custom analyzers when V1 is enabled", async () => {
      const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
      getSettingsSpy.mockResolvedValue({
        ...MOCK_DEFAULT_USER_SETTINGS,
        confirmation_mode: true,
        security_analyzer: "llm",
        v1_enabled: true,
      });

      const getSecurityAnalyzersSpy = vi.spyOn(
        OptionService,
        "getSecurityAnalyzers",
      );
      getSecurityAnalyzersSpy.mockResolvedValue([
        "llm",
        "none",
        "invariant",
        "custom",
      ]);

      renderLlmSettingsScreen();
      await screen.findByTestId("llm-settings-screen");

      const advancedSwitch = screen.getByTestId("advanced-settings-switch");
      await userEvent.click(advancedSwitch);

      const securityAnalyzer = await screen.findByTestId(
        "security-analyzer-input",
      );
      await userEvent.click(securityAnalyzer);

      // Only llm + none should be available when V1 is enabled
      screen.getByText("SETTINGS$SECURITY_ANALYZER_LLM_DEFAULT");
      screen.getByText("SETTINGS$SECURITY_ANALYZER_NONE");
      expect(
        screen.queryByText("SETTINGS$SECURITY_ANALYZER_INVARIANT"),
      ).not.toBeInTheDocument();
      expect(screen.queryByText("custom")).not.toBeInTheDocument();
    });

    it("should include invariant analyzer option when V1 is disabled", async () => {
      const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
      getSettingsSpy.mockResolvedValue({
        ...MOCK_DEFAULT_USER_SETTINGS,
        confirmation_mode: true,
        security_analyzer: "llm",
        v1_enabled: false,
      });

      const getSecurityAnalyzersSpy = vi.spyOn(
        OptionService,
        "getSecurityAnalyzers",
      );
      getSecurityAnalyzersSpy.mockResolvedValue(["llm", "none", "invariant"]);

      renderLlmSettingsScreen();
      await screen.findByTestId("llm-settings-screen");

      const advancedSwitch = screen.getByTestId("advanced-settings-switch");
      await userEvent.click(advancedSwitch);

      const securityAnalyzer = await screen.findByTestId(
        "security-analyzer-input",
      );
      await userEvent.click(securityAnalyzer);

      expect(
        screen.getByText("SETTINGS$SECURITY_ANALYZER_LLM_DEFAULT"),
      ).toBeInTheDocument();
      expect(
        screen.getByText("SETTINGS$SECURITY_ANALYZER_NONE"),
      ).toBeInTheDocument();
      expect(
        screen.getByText("SETTINGS$SECURITY_ANALYZER_INVARIANT"),
      ).toBeInTheDocument();
    });
  });

  it.todo("should render an indicator if the llm api key is set");

  describe("API key visibility in Basic Settings", () => {
    it("should hide API key input when SaaS mode is enabled and OpenHands provider is selected", async () => {
      const getConfigSpy = vi.spyOn(OptionService, "getConfig");
      // @ts-expect-error - only return APP_MODE for these tests
      getConfigSpy.mockResolvedValue({
        APP_MODE: "saas",
      });

      renderLlmSettingsScreen();
      await screen.findByTestId("llm-settings-screen");

      const basicForm = screen.getByTestId("llm-settings-form-basic");
      const provider = within(basicForm).getByTestId("llm-provider-input");

      // Verify OpenHands is selected by default
      await waitFor(() => {
        expect(provider).toHaveValue("OpenHands");
      });

      // API key input should not be visible when OpenHands provider is selected in SaaS mode
      expect(
        within(basicForm).queryByTestId("llm-api-key-input"),
      ).not.toBeInTheDocument();
      expect(
        within(basicForm).queryByTestId("llm-api-key-help-anchor"),
      ).not.toBeInTheDocument();
    });

    it("should show API key input when SaaS mode is enabled and non-OpenHands provider is selected", async () => {
      const getConfigSpy = vi.spyOn(OptionService, "getConfig");
      // @ts-expect-error - only return APP_MODE for these tests
      getConfigSpy.mockResolvedValue({
        APP_MODE: "saas",
      });

      renderLlmSettingsScreen();
      await screen.findByTestId("llm-settings-screen");

      const basicForm = screen.getByTestId("llm-settings-form-basic");
      const provider = within(basicForm).getByTestId("llm-provider-input");

      // Select OpenAI provider
      await userEvent.click(provider);
      const providerOption = screen.getByText("OpenAI");
      await userEvent.click(providerOption);

      await waitFor(() => {
        expect(provider).toHaveValue("OpenAI");
      });

      // API key input should be visible when non-OpenHands provider is selected in SaaS mode
      expect(
        within(basicForm).getByTestId("llm-api-key-input"),
      ).toBeInTheDocument();
      expect(
        within(basicForm).getByTestId("llm-api-key-help-anchor"),
      ).toBeInTheDocument();
    });

    it("should show API key input when OSS mode is enabled and OpenHands provider is selected", async () => {
      // Use OSS mode
      mockUseConfig.mockReturnValue({
        data: { APP_MODE: "oss" },
        isLoading: false,
      });

      renderLlmSettingsScreen();
      await screen.findByTestId("llm-settings-screen");

      const basicForm = screen.getByTestId("llm-settings-form-basic");
      const provider = within(basicForm).getByTestId("llm-provider-input");

      // Verify OpenHands is selected by default
      await waitFor(() => {
        expect(provider).toHaveValue("OpenHands");
      });

      // API key input should be visible when OSS mode is enabled (even with OpenHands provider)
      expect(
        within(basicForm).getByTestId("llm-api-key-input"),
      ).toBeInTheDocument();
      expect(
        within(basicForm).getByTestId("llm-api-key-help-anchor"),
      ).toBeInTheDocument();
    });

    it("should show API key input when OSS mode is enabled and non-OpenHands provider is selected", async () => {
      const getConfigSpy = vi.spyOn(OptionService, "getConfig");
      // @ts-expect-error - only return APP_MODE for these tests
      getConfigSpy.mockResolvedValue({
        APP_MODE: "oss",
      });

      renderLlmSettingsScreen();
      await screen.findByTestId("llm-settings-screen");

      const basicForm = screen.getByTestId("llm-settings-form-basic");
      const provider = within(basicForm).getByTestId("llm-provider-input");

      // Select OpenAI provider
      await userEvent.click(provider);
      const providerOption = screen.getByText("OpenAI");
      await userEvent.click(providerOption);

      await waitFor(() => {
        expect(provider).toHaveValue("OpenAI");
      });

      // API key input should be visible when OSS mode is enabled
      expect(
        within(basicForm).getByTestId("llm-api-key-input"),
      ).toBeInTheDocument();
      expect(
        within(basicForm).getByTestId("llm-api-key-help-anchor"),
      ).toBeInTheDocument();
    });

    it("should hide API key input when switching from non-OpenHands to OpenHands provider in SaaS mode", async () => {
      const getConfigSpy = vi.spyOn(OptionService, "getConfig");
      // @ts-expect-error - only return APP_MODE for these tests
      getConfigSpy.mockResolvedValue({
        APP_MODE: "saas",
      });

      renderLlmSettingsScreen();
      await screen.findByTestId("llm-settings-screen");

      const basicForm = screen.getByTestId("llm-settings-form-basic");
      const provider = within(basicForm).getByTestId("llm-provider-input");

      // Start with OpenAI provider
      await userEvent.click(provider);
      const openAIOption = screen.getByText("OpenAI");
      await userEvent.click(openAIOption);

      await waitFor(() => {
        expect(provider).toHaveValue("OpenAI");
      });

      // API key input should be visible with OpenAI
      expect(
        within(basicForm).getByTestId("llm-api-key-input"),
      ).toBeInTheDocument();

      // Switch to OpenHands provider
      await userEvent.click(provider);
      const openHandsOption = screen.getByText("OpenHands");
      await userEvent.click(openHandsOption);

      await waitFor(() => {
        expect(provider).toHaveValue("OpenHands");
      });

      // API key input should now be hidden
      expect(
        within(basicForm).queryByTestId("llm-api-key-input"),
      ).not.toBeInTheDocument();
      expect(
        within(basicForm).queryByTestId("llm-api-key-help-anchor"),
      ).not.toBeInTheDocument();
    });

    it("should show API key input when switching from OpenHands to non-OpenHands provider in SaaS mode", async () => {
      const getConfigSpy = vi.spyOn(OptionService, "getConfig");
      // @ts-expect-error - only return APP_MODE for these tests
      getConfigSpy.mockResolvedValue({
        APP_MODE: "saas",
      });

      renderLlmSettingsScreen();
      await screen.findByTestId("llm-settings-screen");

      const basicForm = screen.getByTestId("llm-settings-form-basic");
      const provider = within(basicForm).getByTestId("llm-provider-input");

      // Verify OpenHands is selected by default
      await waitFor(() => {
        expect(provider).toHaveValue("OpenHands");
      });

      // API key input should be hidden with OpenHands
      expect(
        within(basicForm).queryByTestId("llm-api-key-input"),
      ).not.toBeInTheDocument();

      // Switch to OpenAI provider
      await userEvent.click(provider);
      const openAIOption = screen.getByText("OpenAI");
      await userEvent.click(openAIOption);

      await waitFor(() => {
        expect(provider).toHaveValue("OpenAI");
      });

      // API key input should now be visible
      expect(
        within(basicForm).getByTestId("llm-api-key-input"),
      ).toBeInTheDocument();
      expect(
        within(basicForm).getByTestId("llm-api-key-help-anchor"),
      ).toBeInTheDocument();
    });
  });
});

describe("Form submission", () => {
  it("should submit the basic form with the correct values", async () => {
    const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");

    renderLlmSettingsScreen();
    await screen.findByTestId("llm-settings-screen");

    const provider = screen.getByTestId("llm-provider-input");
    const model = screen.getByTestId("llm-model-input");

    // select provider (switch to OpenAI so API key input becomes visible)
    await userEvent.click(provider);
    const providerOption = screen.getByText("OpenAI");
    await userEvent.click(providerOption);
    await waitFor(() => {
      expect(provider).toHaveValue("OpenAI");
    });

    // enter api key (now visible after switching provider)
    const apiKey = await screen.findByTestId("llm-api-key-input");
    await userEvent.type(apiKey, "test-api-key");

    // select model
    await userEvent.click(model);
    const modelOption = screen.getByText("gpt-4o");
    await userEvent.click(modelOption);
    expect(model).toHaveValue("gpt-4o");

    const submitButton = screen.getByTestId("submit-button");
    await userEvent.click(submitButton);

    expect(saveSettingsSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        llm_model: "openai/gpt-4o",
        llm_api_key: "test-api-key",
      }),
    );
  });

  it("should submit the advanced form with the correct values", async () => {
    // Use OSS mode so agent-input is visible
    mockUseConfig.mockReturnValue({
      data: { APP_MODE: "oss" },
      isLoading: false,
    });

    const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");

    renderLlmSettingsScreen();
    await screen.findByTestId("llm-settings-screen");

    const advancedSwitch = screen.getByTestId("advanced-settings-switch");
    await userEvent.click(advancedSwitch);

    const model = screen.getByTestId("llm-custom-model-input");
    const baseUrl = screen.getByTestId("base-url-input");
    const apiKey = screen.getByTestId("llm-api-key-input");
    const agent = screen.getByTestId("agent-input");
    const confirmation = screen.getByTestId("enable-confirmation-mode-switch");
    const condensor = screen.getByTestId("enable-memory-condenser-switch");

    // enter custom model
    await userEvent.clear(model);
    await userEvent.type(model, "openai/gpt-4o");
    expect(model).toHaveValue("openai/gpt-4o");

    // enter base url
    await userEvent.type(baseUrl, "https://api.openai.com/v1/chat/completions");
    expect(baseUrl).toHaveValue("https://api.openai.com/v1/chat/completions");

    // enter api key
    await userEvent.type(apiKey, "test-api-key");

    // toggle confirmation mode
    await userEvent.click(confirmation);
    expect(confirmation).toBeChecked();

    // toggle memory condensor
    await userEvent.click(condensor);
    expect(condensor).not.toBeChecked();

    // select agent
    await userEvent.click(agent);
    const agentOption = screen.getByText("CoActAgent");
    await userEvent.click(agentOption);
    expect(agent).toHaveValue("CoActAgent");

    // select security analyzer
    const securityAnalyzer = screen.getByTestId("security-analyzer-input");
    await userEvent.click(securityAnalyzer);
    const securityAnalyzerOption = screen.getByText(
      "SETTINGS$SECURITY_ANALYZER_NONE",
    );
    await userEvent.click(securityAnalyzerOption);

    const submitButton = screen.getByTestId("submit-button");
    await userEvent.click(submitButton);

    expect(saveSettingsSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        llm_model: "openai/gpt-4o",
        llm_base_url: "https://api.openai.com/v1/chat/completions",
        agent: "CoActAgent",
        confirmation_mode: true,
        enable_default_condenser: false,
        security_analyzer: null,
      }),
    );
  });

  it("should disable the button if there are no changes in the basic form", async () => {
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      llm_model: "openai/gpt-4o",
      llm_api_key_set: true,
    });

    renderLlmSettingsScreen();
    await screen.findByTestId("llm-settings-screen");
    screen.getByTestId("llm-settings-form-basic");

    const submitButton = screen.getByTestId("submit-button");
    expect(submitButton).toBeDisabled();

    const model = screen.getByTestId("llm-model-input");
    const apiKey = screen.getByTestId("llm-api-key-input");

    // select model
    await userEvent.click(model);
    const modelOption = screen.getByText("gpt-4o-mini");
    await userEvent.click(modelOption);
    expect(model).toHaveValue("gpt-4o-mini");
    expect(submitButton).not.toBeDisabled();

    // reset model
    await userEvent.click(model);
    const modelOption2 = screen.getByText("gpt-4o");
    await userEvent.click(modelOption2);
    expect(model).toHaveValue("gpt-4o");
    expect(submitButton).toBeDisabled();

    // set api key
    await userEvent.type(apiKey, "test-api-key");
    expect(apiKey).toHaveValue("test-api-key");
    expect(submitButton).not.toBeDisabled();

    // reset api key
    await userEvent.clear(apiKey);
    expect(apiKey).toHaveValue("");
    expect(submitButton).toBeDisabled();
  });

  it("should disable the button if there are no changes in the advanced form", async () => {
    // Use OSS mode so agent-input is visible
    mockUseConfig.mockReturnValue({
      data: { APP_MODE: "oss" },
      isLoading: false,
    });

    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      llm_model: "openai/gpt-4o",
      llm_base_url: "https://api.openai.com/v1/chat/completions",
      llm_api_key_set: true,
      confirmation_mode: true,
    });

    renderLlmSettingsScreen();
    await screen.findByTestId("llm-settings-screen");
    await screen.findByTestId("llm-settings-form-advanced");

    const submitButton = await screen.findByTestId("submit-button");
    expect(submitButton).toBeDisabled();

    const model = await screen.findByTestId("llm-custom-model-input");
    const baseUrl = await screen.findByTestId("base-url-input");
    const apiKey = await screen.findByTestId("llm-api-key-input");
    const agent = await screen.findByTestId("agent-input");
    const condensor = await screen.findByTestId(
      "enable-memory-condenser-switch",
    );

    // Confirmation mode switch is now in basic settings, always visible
    const confirmation = await screen.findByTestId(
      "enable-confirmation-mode-switch",
    );

    // enter custom model
    await userEvent.type(model, "-mini");
    expect(model).toHaveValue("openai/gpt-4o-mini");
    expect(submitButton).not.toBeDisabled();

    // reset model
    await userEvent.clear(model);
    expect(model).toHaveValue("");
    expect(submitButton).toBeDisabled();

    await userEvent.type(model, "openai/gpt-4o");
    expect(model).toHaveValue("openai/gpt-4o");
    expect(submitButton).toBeDisabled();

    // enter base url
    await userEvent.type(baseUrl, "/extra");
    expect(baseUrl).toHaveValue(
      "https://api.openai.com/v1/chat/completions/extra",
    );
    expect(submitButton).not.toBeDisabled();

    await userEvent.clear(baseUrl);
    expect(baseUrl).toHaveValue("");
    expect(submitButton).not.toBeDisabled();

    await userEvent.type(baseUrl, "https://api.openai.com/v1/chat/completions");
    expect(baseUrl).toHaveValue("https://api.openai.com/v1/chat/completions");
    expect(submitButton).toBeDisabled();

    // set api key
    await userEvent.type(apiKey, "test-api-key");
    expect(apiKey).toHaveValue("test-api-key");
    expect(submitButton).not.toBeDisabled();

    // reset api key
    await userEvent.clear(apiKey);
    expect(apiKey).toHaveValue("");
    expect(submitButton).toBeDisabled();

    // set agent
    await userEvent.clear(agent);
    await userEvent.type(agent, "test-agent");
    expect(agent).toHaveValue("test-agent");
    expect(submitButton).not.toBeDisabled();

    // reset agent
    await userEvent.clear(agent);
    expect(agent).toHaveValue("");
    expect(submitButton).toBeDisabled();

    await userEvent.type(agent, "CodeActAgent");
    expect(agent).toHaveValue("CodeActAgent");
    expect(submitButton).toBeDisabled();

    // toggle confirmation mode
    await userEvent.click(confirmation);
    expect(confirmation).not.toBeChecked();
    expect(submitButton).not.toBeDisabled();
    await userEvent.click(confirmation);
    expect(confirmation).toBeChecked();
    expect(submitButton).toBeDisabled();

    // toggle memory condensor
    await userEvent.click(condensor);
    expect(condensor).not.toBeChecked();
    expect(submitButton).not.toBeDisabled();
    await userEvent.click(condensor);
    expect(condensor).toBeChecked();
    expect(submitButton).toBeDisabled();

    // select security analyzer
    const securityAnalyzer = await screen.findByTestId(
      "security-analyzer-input",
    );
    await userEvent.click(securityAnalyzer);
    const securityAnalyzerOption = screen.getByText(
      "SETTINGS$SECURITY_ANALYZER_NONE",
    );
    await userEvent.click(securityAnalyzerOption);
    expect(securityAnalyzer).toHaveValue("SETTINGS$SECURITY_ANALYZER_NONE");

    expect(submitButton).not.toBeDisabled();

    // revert back to original value
    await userEvent.click(securityAnalyzer);
    const originalSecurityAnalyzerOption = screen.getByText(
      "SETTINGS$SECURITY_ANALYZER_LLM_DEFAULT",
    );
    await userEvent.click(originalSecurityAnalyzerOption);
    expect(securityAnalyzer).toHaveValue(
      "SETTINGS$SECURITY_ANALYZER_LLM_DEFAULT",
    );
    expect(submitButton).toBeDisabled();
  });

  it("should reset button state when switching between forms", async () => {
    renderLlmSettingsScreen();
    await screen.findByTestId("llm-settings-screen");

    const advancedSwitch = screen.getByTestId("advanced-settings-switch");
    const submitButton = screen.getByTestId("submit-button");

    expect(submitButton).toBeDisabled();

    // Switch to a non-OpenHands provider first so API key input is visible
    const provider = screen.getByTestId("llm-provider-input");
    await userEvent.click(provider);
    const providerOption = screen.getByText("OpenAI");
    await userEvent.click(providerOption);
    await waitFor(() => {
      expect(provider).toHaveValue("OpenAI");
    });

    // dirty the basic form
    const apiKey = await screen.findByTestId("llm-api-key-input");
    await userEvent.type(apiKey, "test-api-key");
    expect(submitButton).not.toBeDisabled();

    await userEvent.click(advancedSwitch);
    expect(submitButton).toBeDisabled();

    // dirty the advanced form
    const model = screen.getByTestId("llm-custom-model-input");
    await userEvent.type(model, "openai/gpt-4o");
    expect(submitButton).not.toBeDisabled();

    await userEvent.click(advancedSwitch);
    expect(submitButton).toBeDisabled();
  });

  // flaky test
  it.skip("should disable the button when submitting changes", async () => {
    const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");

    renderLlmSettingsScreen();
    await screen.findByTestId("llm-settings-screen");

    const apiKey = screen.getByTestId("llm-api-key-input");
    await userEvent.type(apiKey, "test-api-key");

    const submitButton = screen.getByTestId("submit-button");
    await userEvent.click(submitButton);

    expect(saveSettingsSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        llm_api_key: "test-api-key",
      }),
    );

    expect(submitButton).toHaveTextContent("Saving...");
    expect(submitButton).toBeDisabled();

    await waitFor(() => {
      expect(submitButton).toHaveTextContent("Save");
      expect(submitButton).toBeDisabled();
    });
  });

  it("should clear advanced settings when saving basic settings", async () => {
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      llm_model: "openai/gpt-4o",
      llm_base_url: "https://api.openai.com/v1/chat/completions",
      llm_api_key_set: true,
      confirmation_mode: true,
    });
    const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");
    renderLlmSettingsScreen();

    await screen.findByTestId("llm-settings-screen");
    // Component automatically shows advanced view when advanced settings exist
    // Switch to basic view to test clearing advanced settings
    const advancedSwitch = screen.getByTestId("advanced-settings-switch");
    await userEvent.click(advancedSwitch);

    // Now we should be in basic view
    await screen.findByTestId("llm-settings-form-basic");

    const provider = screen.getByTestId("llm-provider-input");
    const model = screen.getByTestId("llm-model-input");

    // select provider
    await userEvent.click(provider);
    const providerOption = screen.getByText("OpenHands");
    await userEvent.click(providerOption);

    // select model
    await userEvent.click(model);
    const modelOption = screen.getByText("claude-sonnet-4-20250514");
    await userEvent.click(modelOption);

    const submitButton = screen.getByTestId("submit-button");
    await userEvent.click(submitButton);

    expect(saveSettingsSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        llm_model: "openhands/claude-sonnet-4-20250514",
        llm_base_url: "",
        confirmation_mode: false, // Confirmation mode is now an advanced setting, should be cleared when saving basic settings
      }),
    );
  });
});

describe("View persistence after saving advanced settings", () => {
  it("should remain on Advanced view after saving when memory condenser is disabled", async () => {
    // Arrange: Start with default settings (basic view)
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
    });
    const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");
    saveSettingsSpy.mockResolvedValue(true);

    renderLlmSettingsScreen();
    await screen.findByTestId("llm-settings-screen");

    // Verify we start in basic view
    expect(screen.getByTestId("llm-settings-form-basic")).toBeInTheDocument();

    // Act: User manually switches to Advanced view
    const advancedSwitch = screen.getByTestId("advanced-settings-switch");
    await userEvent.click(advancedSwitch);
    await screen.findByTestId("llm-settings-form-advanced");

    // User disables memory condenser (advanced-only setting)
    const condenserSwitch = screen.getByTestId(
      "enable-memory-condenser-switch",
    );
    expect(condenserSwitch).toBeChecked();
    await userEvent.click(condenserSwitch);
    expect(condenserSwitch).not.toBeChecked();

    // Mock the updated settings that will be returned after save
    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      enable_default_condenser: false, // Now disabled
    });

    // User saves settings
    const submitButton = screen.getByTestId("submit-button");
    await userEvent.click(submitButton);

    // Assert: View should remain on Advanced after save
    await waitFor(() => {
      expect(
        screen.getByTestId("llm-settings-form-advanced"),
      ).toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-basic"),
      ).not.toBeInTheDocument();
      expect(advancedSwitch).toBeChecked();
    });
  });

  it("should remain on Advanced view after saving when condenser max size is customized", async () => {
    // Arrange: Start with default settings
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
    });
    const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");
    saveSettingsSpy.mockResolvedValue(true);

    renderLlmSettingsScreen();
    await screen.findByTestId("llm-settings-screen");

    // Act: User manually switches to Advanced view
    const advancedSwitch = screen.getByTestId("advanced-settings-switch");
    await userEvent.click(advancedSwitch);
    await screen.findByTestId("llm-settings-form-advanced");

    // User sets custom condenser max size (advanced-only setting)
    const condenserMaxSizeInput = screen.getByTestId(
      "condenser-max-size-input",
    );
    await userEvent.clear(condenserMaxSizeInput);
    await userEvent.type(condenserMaxSizeInput, "200");

    // Mock the updated settings that will be returned after save
    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      condenser_max_size: 200, // Custom value
    });

    // User saves settings
    const submitButton = screen.getByTestId("submit-button");
    await userEvent.click(submitButton);

    // Assert: View should remain on Advanced after save
    await waitFor(() => {
      expect(
        screen.getByTestId("llm-settings-form-advanced"),
      ).toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-basic"),
      ).not.toBeInTheDocument();
      expect(advancedSwitch).toBeChecked();
    });
  });

  it("should remain on Advanced view after saving when search API key is set", async () => {
    // Arrange: Start with default settings (non-SaaS mode to show search API key field)
    mockUseConfig.mockReturnValue({
      data: {
        APP_MODE: "oss",
        GITHUB_CLIENT_ID: "fake-github-client-id",
        POSTHOG_CLIENT_KEY: "fake-posthog-client-key",
        FEATURE_FLAGS: {
          ENABLE_BILLING: false,
          HIDE_LLM_SETTINGS: false,
          HIDE_BILLING: false,
          ENABLE_JIRA: false,
          ENABLE_JIRA_DC: false,
          ENABLE_LINEAR: false,
        },
      },
      isLoading: false,
    });

    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      search_api_key: "", // Default empty value
    });
    const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");
    saveSettingsSpy.mockResolvedValue(true);

    renderLlmSettingsScreen();
    await screen.findByTestId("llm-settings-screen");

    // Act: User manually switches to Advanced view
    const advancedSwitch = screen.getByTestId("advanced-settings-switch");
    await userEvent.click(advancedSwitch);
    await screen.findByTestId("llm-settings-form-advanced");

    // User sets search API key (advanced-only setting)
    const searchApiKeyInput = screen.getByTestId("search-api-key-input");
    await userEvent.type(searchApiKeyInput, "test-search-api-key");

    // Mock the updated settings that will be returned after save
    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      search_api_key: "test-search-api-key", // Now set
    });

    // User saves settings
    const submitButton = screen.getByTestId("submit-button");
    await userEvent.click(submitButton);

    // Assert: View should remain on Advanced after save
    await waitFor(() => {
      expect(
        screen.getByTestId("llm-settings-form-advanced"),
      ).toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-basic"),
      ).not.toBeInTheDocument();
      expect(advancedSwitch).toBeChecked();
    });
  });
});

describe("Status toasts", () => {
  describe("Basic form", () => {
    it("should call displaySuccessToast when the settings are saved", async () => {
      const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");

      const displaySuccessToastSpy = vi.spyOn(
        ToastHandlers,
        "displaySuccessToast",
      );

      renderLlmSettingsScreen();
      await screen.findByTestId("llm-settings-screen");

      // Switch to a non-OpenHands provider so API key input is visible
      const provider = screen.getByTestId("llm-provider-input");
      await userEvent.click(provider);
      const providerOption = screen.getByText("OpenAI");
      await userEvent.click(providerOption);
      await waitFor(() => {
        expect(provider).toHaveValue("OpenAI");
      });

      // Wait for API key input to appear
      const apiKeyInput = await screen.findByTestId("llm-api-key-input");

      // Also change the model to ensure form is dirty
      const model = screen.getByTestId("llm-model-input");
      await userEvent.click(model);
      const modelOption = screen.getByText("gpt-4o");
      await userEvent.click(modelOption);
      await waitFor(() => {
        expect(model).toHaveValue("gpt-4o");
      });

      // Enter API key
      await userEvent.type(apiKeyInput, "test-api-key");

      // Wait for submit button to be enabled
      const submit = await screen.findByTestId("submit-button");
      await waitFor(() => {
        expect(submit).not.toBeDisabled();
      });
      await userEvent.click(submit);

      expect(saveSettingsSpy).toHaveBeenCalled();
      await waitFor(() => expect(displaySuccessToastSpy).toHaveBeenCalled());
    });

    it("should call displayErrorToast when the settings fail to save", async () => {
      const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");

      const displayErrorToastSpy = vi.spyOn(ToastHandlers, "displayErrorToast");

      saveSettingsSpy.mockRejectedValue(new Error("Failed to save settings"));

      renderLlmSettingsScreen();
      await screen.findByTestId("llm-settings-screen");

      // Switch to a non-OpenHands provider so API key input is visible
      const provider = screen.getByTestId("llm-provider-input");
      await userEvent.click(provider);
      const providerOption = screen.getByText("OpenAI");
      await userEvent.click(providerOption);
      await waitFor(() => {
        expect(provider).toHaveValue("OpenAI");
      });

      // Wait for API key input to appear
      const apiKeyInput = await screen.findByTestId("llm-api-key-input");

      // Also change the model to ensure form is dirty
      const model = screen.getByTestId("llm-model-input");
      await userEvent.click(model);
      const modelOption = screen.getByText("gpt-4o");
      await userEvent.click(modelOption);
      await waitFor(() => {
        expect(model).toHaveValue("gpt-4o");
      });

      // Enter API key
      await userEvent.type(apiKeyInput, "test-api-key");

      // Wait for submit button to be enabled
      const submit = await screen.findByTestId("submit-button");
      await waitFor(() => {
        expect(submit).not.toBeDisabled();
      });
      await userEvent.click(submit);

      expect(saveSettingsSpy).toHaveBeenCalled();
      expect(displayErrorToastSpy).toHaveBeenCalled();
    });
  });

  describe("Advanced form", () => {
    it("should call displaySuccessToast when the settings are saved", async () => {
      // Use OSS mode to ensure API key input is visible
      mockUseConfig.mockReturnValue({
        data: { APP_MODE: "oss" },
        isLoading: false,
      });

      const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");

      const displaySuccessToastSpy = vi.spyOn(
        ToastHandlers,
        "displaySuccessToast",
      );

      renderLlmSettingsScreen();
      await screen.findByTestId("llm-settings-screen");

      const advancedSwitch = screen.getByTestId("advanced-settings-switch");
      await userEvent.click(advancedSwitch);
      await screen.findByTestId("llm-settings-form-advanced");

      // Toggle setting to change
      const apiKeyInput = await screen.findByTestId("llm-api-key-input");
      await userEvent.type(apiKeyInput, "test-api-key");

      // Wait for submit button to be enabled
      const submit = await screen.findByTestId("submit-button");
      await waitFor(() => {
        expect(submit).not.toBeDisabled();
      });
      await userEvent.click(submit);

      expect(saveSettingsSpy).toHaveBeenCalled();
      await waitFor(() => expect(displaySuccessToastSpy).toHaveBeenCalled());
    });

    it("should call displayErrorToast when the settings fail to save", async () => {
      // Use OSS mode to ensure API key input is visible
      mockUseConfig.mockReturnValue({
        data: { APP_MODE: "oss" },
        isLoading: false,
      });

      const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");

      const displayErrorToastSpy = vi.spyOn(ToastHandlers, "displayErrorToast");

      saveSettingsSpy.mockRejectedValue(new Error("Failed to save settings"));

      renderLlmSettingsScreen();
      await screen.findByTestId("llm-settings-screen");

      const advancedSwitch = screen.getByTestId("advanced-settings-switch");
      await userEvent.click(advancedSwitch);
      await screen.findByTestId("llm-settings-form-advanced");

      // Toggle setting to change
      const apiKeyInput = await screen.findByTestId("llm-api-key-input");
      await userEvent.type(apiKeyInput, "test-api-key");

      // Wait for submit button to be enabled
      const submit = await screen.findByTestId("submit-button");
      await waitFor(() => {
        expect(submit).not.toBeDisabled();
      });
      await userEvent.click(submit);

      expect(saveSettingsSpy).toHaveBeenCalled();
      expect(displayErrorToastSpy).toHaveBeenCalled();
    });
  });
});

describe("Role-based permissions", () => {
  const getConfigSpy = vi.spyOn(OptionService, "getConfig");
  const getMeSpy = vi.spyOn(organizationService, "getMe");

  beforeEach(() => {
    // @ts-expect-error - only return APP_MODE for these tests
    getConfigSpy.mockResolvedValue({
      APP_MODE: "saas",
    });
  });

  describe("User role (read-only)", () => {
    const memberData: OrganizationMember = {
      org_id: "2",
      user_id: "99",
      email: "user@example.com",
      role: "member",
      status: "active",
      llm_api_key: "",
      max_iterations: 20,
      llm_model: "",
      llm_api_key_for_byor: null,
      llm_base_url: "",
    };

    beforeEach(() => {
      // Mock user role
      getMeSpy.mockResolvedValue(memberData);
    });

    it("should disable all input fields in basic view", async () => {
      // Arrange
      renderLlmSettingsScreen("2", memberData); // orgId "2" returns user role

      // Act
      await screen.findByTestId("llm-settings-screen");
      const basicForm = screen.getByTestId("llm-settings-form-basic");

      // Assert
      const providerInput = within(basicForm).getByTestId("llm-provider-input");
      const modelInput = within(basicForm).getByTestId("llm-model-input");

      await waitFor(() => {
        expect(providerInput).toBeDisabled();
        expect(modelInput).toBeDisabled();
      });

      // API key input may be hidden if OpenHands provider is selected in SaaS mode
      // If it exists, it should be disabled
      const apiKeyInput = within(basicForm).queryByTestId("llm-api-key-input");
      if (apiKeyInput) {
        expect(apiKeyInput).toBeDisabled();
      }
    });

    it("should disable all input fields in advanced view", async () => {
      // Arrange
      renderLlmSettingsScreen("2", memberData);

      // Act
      await screen.findByTestId("llm-settings-screen");
      const advancedSwitch = screen.getByTestId("advanced-settings-switch");
      await userEvent.click(advancedSwitch);
      const advancedForm = await screen.findByTestId(
        "llm-settings-form-advanced",
      );

      // Assert
      const modelInput = within(advancedForm).getByTestId(
        "llm-custom-model-input",
      );
      const baseUrlInput = within(advancedForm).getByTestId("base-url-input");
      const condenserSwitch = within(advancedForm).getByTestId(
        "enable-memory-condenser-switch",
      );
      const confirmationSwitch = within(advancedForm).getByTestId(
        "enable-confirmation-mode-switch",
      );

      await waitFor(() => {
        expect(modelInput).toBeDisabled();
        expect(baseUrlInput).toBeDisabled();
        expect(condenserSwitch).toBeDisabled();
        expect(confirmationSwitch).toBeDisabled();
      });

      // API key input may be hidden if OpenHands provider is selected in SaaS mode
      // If it exists, it should be disabled
      const apiKeyInput =
        within(advancedForm).queryByTestId("llm-api-key-input");
      if (apiKeyInput) {
        expect(apiKeyInput).toBeDisabled();
      }

      // Agent input is only visible in non-SaaS mode and when V1 is not enabled
      // If it exists, it should be disabled
      const agentInput = within(advancedForm).queryByTestId("agent-input");
      if (agentInput) {
        expect(agentInput).toBeDisabled();
      }
    });

    it("should not render submit button", async () => {
      // Arrange
      renderLlmSettingsScreen("2", memberData);

      // Act
      await screen.findByTestId("llm-settings-screen");
      const submitButton = screen.queryByTestId("submit-button");

      // Assert
      expect(submitButton).not.toBeInTheDocument();
    });

    it("should allow toggling between basic and advanced views", async () => {
      // Arrange
      renderLlmSettingsScreen("2", memberData);

      // Act
      await screen.findByTestId("llm-settings-screen");
      const advancedSwitch = screen.getByTestId("advanced-settings-switch");
      const basicForm = screen.getByTestId("llm-settings-form-basic");

      // Assert - toggle should be enabled
      expect(advancedSwitch).not.toBeDisabled();

      // Act - toggle to advanced
      await userEvent.click(advancedSwitch);
      const advancedForm = await screen.findByTestId(
        "llm-settings-form-advanced",
      );

      // Assert - advanced form is visible
      expect(advancedForm).toBeInTheDocument();
      expect(basicForm).not.toBeInTheDocument();

      // Act - toggle back to basic
      await userEvent.click(advancedSwitch);
      const basicFormAgain = await screen.findByTestId(
        "llm-settings-form-basic",
      );

      // Assert - basic form is visible again
      expect(basicFormAgain).toBeInTheDocument();
      expect(
        screen.queryByTestId("llm-settings-form-advanced"),
      ).not.toBeInTheDocument();
    });

    it("should disable security analyzer dropdown when confirmation mode is enabled", async () => {
      // Arrange
      const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
      getSettingsSpy.mockResolvedValue({
        ...MOCK_DEFAULT_USER_SETTINGS,
        confirmation_mode: true,
      });

      renderLlmSettingsScreen("2", memberData);

      // Act
      await screen.findByTestId("llm-settings-screen");
      const advancedSwitch = screen.getByTestId("advanced-settings-switch");
      await userEvent.click(advancedSwitch);
      await screen.findByTestId("llm-settings-form-advanced");

      // Assert
      const securityAnalyzer = screen.getByTestId("security-analyzer-input");
      await waitFor(() => {
        expect(securityAnalyzer).toBeDisabled();
      });
    });
  });

  describe("Owner role (full access)", () => {
    beforeEach(() => {
      // Mock owner role
      getMeSpy.mockResolvedValue({
        org_id: "1",
        user_id: "99",
        email: "owner@example.com",
        role: "owner",
        status: "active",
        llm_api_key: "",
        max_iterations: 20,
        llm_model: "",
        llm_api_key_for_byor: null,
        llm_base_url: "",
      });
    });

    it("should enable all input fields in basic view", async () => {
      // Arrange
      renderLlmSettingsScreen("1"); // orgId "1" returns owner role

      // Act
      await screen.findByTestId("llm-settings-screen");
      const basicForm = screen.getByTestId("llm-settings-form-basic");

      // Assert
      const providerInput = within(basicForm).getByTestId("llm-provider-input");
      const modelInput = within(basicForm).getByTestId("llm-model-input");

      await waitFor(() => {
        expect(providerInput).not.toBeDisabled();
        expect(modelInput).not.toBeDisabled();
      });

      // API key input may be hidden if OpenHands provider is selected in SaaS mode
      // If it exists, it should be enabled
      const apiKeyInput = within(basicForm).queryByTestId("llm-api-key-input");
      if (apiKeyInput) {
        expect(apiKeyInput).not.toBeDisabled();
      }
    });

    it("should enable all input fields in advanced view", async () => {
      // Arrange
      renderLlmSettingsScreen("1");

      // Act
      await screen.findByTestId("llm-settings-screen");
      const advancedSwitch = screen.getByTestId("advanced-settings-switch");
      await userEvent.click(advancedSwitch);
      const advancedForm = await screen.findByTestId(
        "llm-settings-form-advanced",
      );

      // Assert
      const modelInput = within(advancedForm).getByTestId(
        "llm-custom-model-input",
      );
      const baseUrlInput = within(advancedForm).getByTestId("base-url-input");
      const condenserSwitch = within(advancedForm).getByTestId(
        "enable-memory-condenser-switch",
      );
      const confirmationSwitch = within(advancedForm).getByTestId(
        "enable-confirmation-mode-switch",
      );

      await waitFor(() => {
        expect(modelInput).not.toBeDisabled();
        expect(baseUrlInput).not.toBeDisabled();
        expect(condenserSwitch).not.toBeDisabled();
        expect(confirmationSwitch).not.toBeDisabled();
      });

      // API key input may be hidden if OpenHands provider is selected in SaaS mode
      // If it exists, it should be enabled
      const apiKeyInput =
        within(advancedForm).queryByTestId("llm-api-key-input");
      if (apiKeyInput) {
        expect(apiKeyInput).not.toBeDisabled();
      }
    });

    it("should enable submit button when form is dirty", async () => {
      // Arrange
      renderLlmSettingsScreen("1");

      // Act
      await screen.findByTestId("llm-settings-screen");
      const submitButton = screen.getByTestId("submit-button");
      const providerInput = screen.getByTestId("llm-provider-input");

      // Assert - initially disabled (no changes)
      expect(submitButton).toBeDisabled();

      // Act - make a change by selecting a different provider
      await userEvent.click(providerInput);
      const openAIOption = await screen.findByText("OpenAI");
      await userEvent.click(openAIOption);

      // Assert - button should be enabled
      await waitFor(() => {
        expect(submitButton).not.toBeDisabled();
      });
    });

    it("should allow submitting form changes", async () => {
      // Arrange
      const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");
      renderLlmSettingsScreen("1");

      // Act
      await screen.findByTestId("llm-settings-screen");
      const providerInput = screen.getByTestId("llm-provider-input");
      const modelInput = screen.getByTestId("llm-model-input");

      // Select a different provider to make form dirty
      await userEvent.click(providerInput);
      const openAIOption = await screen.findByText("OpenAI");
      await userEvent.click(openAIOption);
      await waitFor(() => {
        expect(providerInput).toHaveValue("OpenAI");
      });

      // Select a different model to ensure form is dirty
      await userEvent.click(modelInput);
      const modelOption = await screen.findByText("gpt-4o");
      await userEvent.click(modelOption);
      await waitFor(() => {
        expect(modelInput).toHaveValue("gpt-4o");
      });

      // Wait for form to be marked as dirty
      const submitButton = await screen.findByTestId("submit-button");
      await waitFor(() => {
        expect(submitButton).not.toBeDisabled();
      });

      await userEvent.click(submitButton);

      // Assert
      await waitFor(() => {
        expect(saveSettingsSpy).toHaveBeenCalled();
      });
    });
  });

  describe("Admin role (full access)", () => {
    beforeEach(() => {
      // Mock admin role
      getMeSpy.mockResolvedValue({
        org_id: "3",
        user_id: "99",
        email: "admin@example.com",
        role: "admin",
        status: "active",
        llm_api_key: "",
        max_iterations: 20,
        llm_model: "",
        llm_api_key_for_byor: null,
        llm_base_url: "",
      });
    });

    it("should enable all input fields in basic view", async () => {
      // Arrange
      renderLlmSettingsScreen("3"); // orgId "3" returns admin role

      // Act
      await screen.findByTestId("llm-settings-screen");
      const basicForm = screen.getByTestId("llm-settings-form-basic");

      // Assert
      const providerInput = within(basicForm).getByTestId("llm-provider-input");
      const modelInput = within(basicForm).getByTestId("llm-model-input");

      await waitFor(() => {
        expect(providerInput).not.toBeDisabled();
        expect(modelInput).not.toBeDisabled();
      });

      // API key input may be hidden if OpenHands provider is selected in SaaS mode
      // If it exists, it should be enabled
      const apiKeyInput = within(basicForm).queryByTestId("llm-api-key-input");
      if (apiKeyInput) {
        expect(apiKeyInput).not.toBeDisabled();
      }
    });

    it("should enable all input fields in advanced view", async () => {
      // Arrange
      renderLlmSettingsScreen("3");

      // Act
      await screen.findByTestId("llm-settings-screen");
      const advancedSwitch = screen.getByTestId("advanced-settings-switch");
      await userEvent.click(advancedSwitch);
      const advancedForm = await screen.findByTestId(
        "llm-settings-form-advanced",
      );

      // Assert
      const modelInput = within(advancedForm).getByTestId(
        "llm-custom-model-input",
      );
      const baseUrlInput = within(advancedForm).getByTestId("base-url-input");
      const condenserSwitch = within(advancedForm).getByTestId(
        "enable-memory-condenser-switch",
      );
      const confirmationSwitch = within(advancedForm).getByTestId(
        "enable-confirmation-mode-switch",
      );

      await waitFor(() => {
        expect(modelInput).not.toBeDisabled();
        expect(baseUrlInput).not.toBeDisabled();
        expect(condenserSwitch).not.toBeDisabled();
        expect(confirmationSwitch).not.toBeDisabled();
      });

      // API key input may be hidden if OpenHands provider is selected in SaaS mode
      // If it exists, it should be enabled
      const apiKeyInput =
        within(advancedForm).queryByTestId("llm-api-key-input");
      if (apiKeyInput) {
        expect(apiKeyInput).not.toBeDisabled();
      }
    });

    it("should enable submit button when form is dirty", async () => {
      // Arrange
      renderLlmSettingsScreen("3");

      // Act
      await screen.findByTestId("llm-settings-screen");
      const submitButton = screen.getByTestId("submit-button");
      const providerInput = screen.getByTestId("llm-provider-input");

      // Assert - initially disabled (no changes)
      expect(submitButton).toBeDisabled();

      // Act - make a change by selecting a different provider
      await userEvent.click(providerInput);
      const openAIOption = await screen.findByText("OpenAI");
      await userEvent.click(openAIOption);

      // Assert - button should be enabled
      await waitFor(() => {
        expect(submitButton).not.toBeDisabled();
      });
    });

    it("should allow submitting form changes", async () => {
      // Arrange
      const saveSettingsSpy = vi.spyOn(SettingsService, "saveSettings");
      renderLlmSettingsScreen("3");

      // Act
      await screen.findByTestId("llm-settings-screen");
      const providerInput = screen.getByTestId("llm-provider-input");
      const modelInput = screen.getByTestId("llm-model-input");

      // Select a different provider to make form dirty
      await userEvent.click(providerInput);
      const openAIOption = await screen.findByText("OpenAI");
      await userEvent.click(openAIOption);
      await waitFor(() => {
        expect(providerInput).toHaveValue("OpenAI");
      });

      // Select a different model to ensure form is dirty
      await userEvent.click(modelInput);
      const modelOption = await screen.findByText("gpt-4o");
      await userEvent.click(modelOption);
      await waitFor(() => {
        expect(modelInput).toHaveValue("gpt-4o");
      });

      // Wait for form to be marked as dirty
      const submitButton = await screen.findByTestId("submit-button");
      await waitFor(() => {
        expect(submitButton).not.toBeDisabled();
      });

      await userEvent.click(submitButton);

      // Assert
      await waitFor(() => {
        expect(saveSettingsSpy).toHaveBeenCalled();
      });
    });
  });
});
