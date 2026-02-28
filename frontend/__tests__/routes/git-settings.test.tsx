import { render, screen, waitFor } from "@testing-library/react";
import { createRoutesStub } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import i18next from "i18next";
import { I18nextProvider } from "react-i18next";
import GitSettingsScreen from "#/routes/git-settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import OptionService from "#/api/option-service/option-service.api";
import AuthService from "#/api/auth-service/auth-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { WebClientConfig } from "#/api/option-service/option.types";
import * as ToastHandlers from "#/utils/custom-toast-handlers";
import { SecretsService } from "#/api/secrets-service";
import { integrationService } from "#/api/integration-service/integration-service.api";

const VALID_OSS_CONFIG: WebClientConfig = {
  app_mode: "oss",
  posthog_client_key: "456",
  feature_flags: {
    enable_billing: false,
    hide_llm_settings: false,
    enable_jira: false,
    enable_jira_dc: false,
    enable_linear: false,
  },
  providers_configured: [],
  maintenance_start_time: null,
  auth_url: null,
  recaptcha_site_key: null,
  faulty_models: [],
  error_message: null,
  updated_at: "2024-01-14T10:00:00Z",
  github_app_slug: null,
};

const VALID_SAAS_CONFIG: WebClientConfig = {
  app_mode: "saas",
  posthog_client_key: "456",
  feature_flags: {
    enable_billing: false,
    hide_llm_settings: false,
    enable_jira: false,
    enable_jira_dc: false,
    enable_linear: false,
  },
  providers_configured: [],
  maintenance_start_time: null,
  auth_url: null,
  recaptcha_site_key: null,
  faulty_models: [],
  error_message: null,
  updated_at: "2024-01-14T10:00:00Z",
  github_app_slug: null,
};

const queryClient = new QueryClient();

const GitSettingsRouterStub = createRoutesStub([
  {
    Component: GitSettingsScreen,
    path: "/settings/integrations",
  },
]);

const renderGitSettingsScreen = () => {
  // Initialize i18next instance
  i18next.init({
    lng: "en",
    resources: {
      en: {
        translation: {
          GITHUB$TOKEN_HELP_TEXT: "Help text",
          GITHUB$TOKEN_LABEL: "GitHub Token",
          GITHUB$HOST_LABEL: "GitHub Host",
          GITLAB$TOKEN_LABEL: "GitLab Token",
          GITLAB$HOST_LABEL: "GitLab Host",
          BITBUCKET$TOKEN_LABEL: "Bitbucket Token",
          BITBUCKET$HOST_LABEL: "Bitbucket Host",
          SETTINGS$GITLAB: "GitLab",
          COMMON$STATUS: "Status",
          STATUS$CONNECTED: "Connected",
          SETTINGS$GITLAB_NOT_CONNECTED: "Not Connected",
          SETTINGS$GITLAB_REINSTALL_WEBHOOK: "Reinstall Webhook",
          SETTINGS$GITLAB_INSTALLING_WEBHOOK:
            "Installing GitLab webhook, please wait a few minutes.",
          SETTINGS$SAVING: "Saving...",
          ERROR$GENERIC: "An error occurred",
        },
      },
    },
  });

  const { rerender, ...rest } = render(
    <GitSettingsRouterStub initialEntries={["/settings/integrations"]} />,
    {
      wrapper: ({ children }) => (
        <I18nextProvider i18n={i18next}>
          <QueryClientProvider client={queryClient}>
            {children}
          </QueryClientProvider>
        </I18nextProvider>
      ),
    },
  );

  const rerenderGitSettingsScreen = () =>
    rerender(
      <I18nextProvider i18n={i18next}>
        <QueryClientProvider client={queryClient}>
          <GitSettingsRouterStub initialEntries={["/settings/integrations"]} />
        </QueryClientProvider>
      </I18nextProvider>,
    );

  return {
    ...rest,
    rerender: rerenderGitSettingsScreen,
  };
};

beforeEach(() => {
  // Since we don't recreate the query client on every test, we need to
  // reset the query client before each test to avoid state leaks
  // between tests.
  queryClient.invalidateQueries();
});

describe("Content", () => {
  it("should render", async () => {
    renderGitSettingsScreen();
    await screen.findByTestId("git-settings-screen");
  });

  it("should render the inputs if OSS mode", async () => {
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    getConfigSpy.mockResolvedValue(VALID_OSS_CONFIG);

    const { rerender } = renderGitSettingsScreen();

    await screen.findByTestId("github-token-input");
    await screen.findByTestId("github-token-help-anchor");

    await screen.findByTestId("gitlab-token-input");
    await screen.findByTestId("gitlab-token-help-anchor");

    await screen.findByTestId("bitbucket-token-input");
    await screen.findByTestId("bitbucket-token-help-anchor");

    await screen.findByTestId("azure-devops-token-input");
    await screen.findByTestId("azure-devops-token-help-anchor");

    getConfigSpy.mockResolvedValue(VALID_SAAS_CONFIG);
    queryClient.invalidateQueries();
    rerender();

    await waitFor(() => {
      expect(
        screen.queryByTestId("github-token-input"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByTestId("github-token-help-anchor"),
      ).not.toBeInTheDocument();

      expect(
        screen.queryByTestId("gitlab-token-input"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByTestId("gitlab-token-help-anchor"),
      ).not.toBeInTheDocument();

      expect(
        screen.queryByTestId("bitbucket-token-input"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByTestId("bitbucket-token-help-anchor"),
      ).not.toBeInTheDocument();

      expect(
        screen.queryByTestId("azure-devops-token-input"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByTestId("azure-devops-token-help-anchor"),
      ).not.toBeInTheDocument();
    });
  });

  it("should set '<hidden>' placeholder and indicator if the GitHub token is set", async () => {
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");

    getConfigSpy.mockResolvedValue(VALID_OSS_CONFIG);
    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
    });

    const { rerender } = renderGitSettingsScreen();

    await waitFor(() => {
      const githubInput = screen.getByTestId("github-token-input");
      expect(githubInput).toHaveProperty("placeholder", "");
      expect(
        screen.queryByTestId("gh-set-token-indicator"),
      ).not.toBeInTheDocument();

      const gitlabInput = screen.getByTestId("gitlab-token-input");
      expect(gitlabInput).toHaveProperty("placeholder", "");
      expect(
        screen.queryByTestId("gl-set-token-indicator"),
      ).not.toBeInTheDocument();
    });

    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      provider_tokens_set: {
        github: null,
        gitlab: null,
      },
    });
    queryClient.invalidateQueries();

    rerender();

    await waitFor(() => {
      const githubInput = screen.getByTestId("github-token-input");
      expect(githubInput).toHaveProperty("placeholder", "<hidden>");
      expect(
        screen.queryByTestId("gh-set-token-indicator"),
      ).toBeInTheDocument();

      const gitlabInput = screen.getByTestId("gitlab-token-input");
      expect(gitlabInput).toHaveProperty("placeholder", "<hidden>");
      expect(
        screen.queryByTestId("gl-set-token-indicator"),
      ).toBeInTheDocument();
    });

    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      provider_tokens_set: {
        gitlab: null,
      },
    });
    queryClient.invalidateQueries();

    rerender();

    await waitFor(() => {
      const githubInput = screen.getByTestId("github-token-input");
      expect(githubInput).toHaveProperty("placeholder", "");
      expect(
        screen.queryByTestId("gh-set-token-indicator"),
      ).not.toBeInTheDocument();

      const gitlabInput = screen.getByTestId("gitlab-token-input");
      expect(gitlabInput).toHaveProperty("placeholder", "<hidden>");
      expect(
        screen.queryByTestId("gl-set-token-indicator"),
      ).toBeInTheDocument();
    });
  });

  it("should render the 'Configure GitHub Repositories' button if SaaS mode and github_app_slug exists", async () => {
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    getConfigSpy.mockResolvedValue(VALID_OSS_CONFIG);

    const { rerender } = renderGitSettingsScreen();

    let button = screen.queryByTestId("configure-github-repositories-button");
    expect(button).not.toBeInTheDocument();

    expect(screen.getByTestId("submit-button")).toBeInTheDocument();
    expect(screen.getByTestId("disconnect-tokens-button")).toBeInTheDocument();

    getConfigSpy.mockResolvedValue(VALID_SAAS_CONFIG);
    queryClient.invalidateQueries();
    rerender();

    await waitFor(() => {
      // wait until queries are resolved
      expect(queryClient.isFetching()).toBe(0);
      button = screen.queryByTestId("configure-github-repositories-button");
      expect(button).not.toBeInTheDocument();
    });

    getConfigSpy.mockResolvedValue({
      ...VALID_SAAS_CONFIG,
      github_app_slug: "test-slug",
    });
    queryClient.invalidateQueries();
    rerender();

    await waitFor(() => {
      button = screen.getByTestId("configure-github-repositories-button");
      expect(button).toBeInTheDocument();
      expect(screen.queryByTestId("submit-button")).not.toBeInTheDocument();
      expect(
        screen.queryByTestId("disconnect-tokens-button"),
      ).not.toBeInTheDocument();
    });
  });
});

describe("Form submission", () => {
  it("should save the GitHub token", async () => {
    const saveProvidersSpy = vi.spyOn(SecretsService, "addGitProvider");
    saveProvidersSpy.mockImplementation(() => Promise.resolve(true));
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    getConfigSpy.mockResolvedValue(VALID_OSS_CONFIG);

    renderGitSettingsScreen();

    const githubInput = await screen.findByTestId("github-token-input");
    const submit = await screen.findByTestId("submit-button");

    await userEvent.type(githubInput, "test-token");
    await userEvent.click(submit);

    expect(saveProvidersSpy).toHaveBeenCalledWith({
      github: { token: "test-token", host: "" },
      gitlab: { token: "", host: "" },
      bitbucket: { token: "", host: "" },
      azure_devops: { token: "", host: "" },
      forgejo: { token: "", host: "" },
    });
  });

  it("should save GitLab tokens", async () => {
    const saveProvidersSpy = vi.spyOn(SecretsService, "addGitProvider");
    saveProvidersSpy.mockImplementation(() => Promise.resolve(true));
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    getConfigSpy.mockResolvedValue(VALID_OSS_CONFIG);

    renderGitSettingsScreen();

    const gitlabInput = await screen.findByTestId("gitlab-token-input");
    const submit = await screen.findByTestId("submit-button");

    await userEvent.type(gitlabInput, "test-token");
    await userEvent.click(submit);

    expect(saveProvidersSpy).toHaveBeenCalledWith({
      github: { token: "", host: "" },
      gitlab: { token: "test-token", host: "" },
      bitbucket: { token: "", host: "" },
      azure_devops: { token: "", host: "" },
      forgejo: { token: "", host: "" },
    });
  });

  it("should save the Bitbucket token", async () => {
    const saveProvidersSpy = vi.spyOn(SecretsService, "addGitProvider");
    saveProvidersSpy.mockImplementation(() => Promise.resolve(true));
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    getConfigSpy.mockResolvedValue(VALID_OSS_CONFIG);

    renderGitSettingsScreen();

    const bitbucketInput = await screen.findByTestId("bitbucket-token-input");
    const submit = await screen.findByTestId("submit-button");

    await userEvent.type(bitbucketInput, "test-token");
    await userEvent.click(submit);

    expect(saveProvidersSpy).toHaveBeenCalledWith({
      github: { token: "", host: "" },
      gitlab: { token: "", host: "" },
      bitbucket: { token: "test-token", host: "" },
      azure_devops: { token: "", host: "" },
      forgejo: { token: "", host: "" },
    });
  });

  it("should save the Azure DevOps token", async () => {
    const saveProvidersSpy = vi.spyOn(SecretsService, "addGitProvider");
    saveProvidersSpy.mockImplementation(() => Promise.resolve(true));
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    getConfigSpy.mockResolvedValue(VALID_OSS_CONFIG);

    renderGitSettingsScreen();

    const azureDevOpsInput = await screen.findByTestId(
      "azure-devops-token-input",
    );
    const submit = await screen.findByTestId("submit-button");

    await userEvent.type(azureDevOpsInput, "test-token");
    await userEvent.click(submit);

    expect(saveProvidersSpy).toHaveBeenCalledWith({
      github: { token: "", host: "" },
      gitlab: { token: "", host: "" },
      bitbucket: { token: "", host: "" },
      azure_devops: { token: "test-token", host: "" },
      forgejo: { token: "", host: "" },
    });
  });

  it("should disable the button if there is no input", async () => {
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    getConfigSpy.mockResolvedValue(VALID_OSS_CONFIG);

    renderGitSettingsScreen();

    const submit = await screen.findByTestId("submit-button");
    expect(submit).toBeDisabled();

    const githubInput = await screen.findByTestId("github-token-input");
    await userEvent.type(githubInput, "test-token");

    expect(submit).not.toBeDisabled();

    await userEvent.clear(githubInput);
    expect(submit).toBeDisabled();

    const gitlabInput = await screen.findByTestId("gitlab-token-input");
    await userEvent.type(gitlabInput, "test-token");

    expect(submit).not.toBeDisabled();

    await userEvent.clear(gitlabInput);
    expect(submit).toBeDisabled();
  });

  it("should enable a disconnect tokens button if there is at least one token set", async () => {
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");

    getConfigSpy.mockResolvedValue(VALID_OSS_CONFIG);
    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      provider_tokens_set: {
        github: null,
        gitlab: null,
      },
    });

    renderGitSettingsScreen();
    await screen.findByTestId("git-settings-screen");

    let disconnectButton = await screen.findByTestId(
      "disconnect-tokens-button",
    );
    // When tokens are set (github and gitlab are not null), the button should be enabled
    await waitFor(() => expect(disconnectButton).not.toBeDisabled());

    // Mock settings with no tokens set
    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      provider_tokens_set: {},
    });
    queryClient.invalidateQueries();

    disconnectButton = await screen.findByTestId("disconnect-tokens-button");
    // When no tokens are set, the button should be disabled
    await waitFor(() => expect(disconnectButton).toBeDisabled());
  });

  it("should call logout when pressing the disconnect tokens button", async () => {
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    const logoutSpy = vi.spyOn(AuthService, "logout");
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");

    getConfigSpy.mockResolvedValue(VALID_OSS_CONFIG);
    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      provider_tokens_set: {
        github: null,
        gitlab: null,
      },
    });

    renderGitSettingsScreen();

    const disconnectButton = await screen.findByTestId(
      "disconnect-tokens-button",
    );
    await waitFor(() => expect(disconnectButton).not.toBeDisabled());
    await userEvent.click(disconnectButton);

    expect(logoutSpy).toHaveBeenCalled();
  });

  // flaky test
  it.skip("should disable the button when submitting changes", async () => {
    const saveSettingsSpy = vi.spyOn(SecretsService, "addGitProvider");
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    getConfigSpy.mockResolvedValue(VALID_OSS_CONFIG);

    renderGitSettingsScreen();

    const submit = await screen.findByTestId("submit-button");
    expect(submit).toBeDisabled();

    const githubInput = await screen.findByTestId("github-token-input");
    await userEvent.type(githubInput, "test-token");
    expect(submit).not.toBeDisabled();

    // submit the form
    await userEvent.click(submit);
    expect(saveSettingsSpy).toHaveBeenCalled();

    expect(submit).toHaveTextContent("Saving...");
    expect(submit).toBeDisabled();

    await waitFor(() => expect(submit).toHaveTextContent("Save"));
  });

  it("should disable the button after submitting changes", async () => {
    const saveProvidersSpy = vi.spyOn(SecretsService, "addGitProvider");
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    getConfigSpy.mockResolvedValue(VALID_OSS_CONFIG);

    renderGitSettingsScreen();
    await screen.findByTestId("git-settings-screen");

    const submit = await screen.findByTestId("submit-button");
    expect(submit).toBeDisabled();

    const githubInput = await screen.findByTestId("github-token-input");
    await userEvent.type(githubInput, "test-token");
    expect(submit).not.toBeDisabled();

    // submit the form
    await userEvent.click(submit);
    expect(saveProvidersSpy).toHaveBeenCalled();
    expect(submit).toBeDisabled();

    const gitlabInput = await screen.findByTestId("gitlab-token-input");
    await userEvent.type(gitlabInput, "test-token");
    expect(gitlabInput).toHaveValue("test-token");
    expect(submit).not.toBeDisabled();

    // submit the form
    await userEvent.click(submit);
    expect(saveProvidersSpy).toHaveBeenCalled();

    await waitFor(() => expect(submit).toBeDisabled());
  });
});

describe("Status toasts", () => {
  it("should call displaySuccessToast when the settings are saved", async () => {
    const saveProvidersSpy = vi.spyOn(SecretsService, "addGitProvider");
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    getSettingsSpy.mockResolvedValue(MOCK_DEFAULT_USER_SETTINGS);

    const displaySuccessToastSpy = vi.spyOn(
      ToastHandlers,
      "displaySuccessToast",
    );

    renderGitSettingsScreen();

    // Toggle setting to change
    const githubInput = await screen.findByTestId("github-token-input");
    await userEvent.type(githubInput, "test-token");

    const submit = await screen.findByTestId("submit-button");
    await userEvent.click(submit);

    expect(saveProvidersSpy).toHaveBeenCalled();
    await waitFor(() => expect(displaySuccessToastSpy).toHaveBeenCalled());
  });

  it("should call displayErrorToast when the settings fail to save", async () => {
    const saveProvidersSpy = vi.spyOn(SecretsService, "addGitProvider");
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
    getSettingsSpy.mockResolvedValue(MOCK_DEFAULT_USER_SETTINGS);

    const displayErrorToastSpy = vi.spyOn(ToastHandlers, "displayErrorToast");

    saveProvidersSpy.mockRejectedValue(new Error("Failed to save settings"));

    renderGitSettingsScreen();

    // Toggle setting to change
    const gitlabInput = await screen.findByTestId("gitlab-token-input");
    await userEvent.type(gitlabInput, "test-token");

    const submit = await screen.findByTestId("submit-button");
    await userEvent.click(submit);

    expect(saveProvidersSpy).toHaveBeenCalled();
    expect(displayErrorToastSpy).toHaveBeenCalled();
  });
});

describe("GitLab Webhook Manager Integration", () => {
  it("should not render GitLab webhook manager in OSS mode", async () => {
    // Arrange
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    getConfigSpy.mockResolvedValue(VALID_OSS_CONFIG);

    // Act
    renderGitSettingsScreen();
    await screen.findByTestId("git-settings-screen");

    // Assert
    await waitFor(() => {
      expect(
        screen.queryByText("GITLAB$WEBHOOK_MANAGER_TITLE"),
      ).not.toBeInTheDocument();
    });
  });

  it("should not render GitLab webhook manager in SaaS mode without APP_SLUG", async () => {
    // Arrange
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    getConfigSpy.mockResolvedValue(VALID_SAAS_CONFIG);

    // Act
    renderGitSettingsScreen();
    await screen.findByTestId("git-settings-screen");

    // Assert
    await waitFor(() => {
      expect(
        screen.queryByText("GITLAB$WEBHOOK_MANAGER_TITLE"),
      ).not.toBeInTheDocument();
    });
  });

  it("should not render GitLab webhook manager when token is not set", async () => {
    // Arrange
    const getConfigSpy = vi.spyOn(OptionService, "getConfig");
    const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");

    getConfigSpy.mockResolvedValue({
      ...VALID_SAAS_CONFIG,
    });
    getSettingsSpy.mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      provider_tokens_set: {},
    });

    // Act
    renderGitSettingsScreen();
    await screen.findByTestId("git-settings-screen");

    // Assert
    await waitFor(() => {
      expect(
        screen.queryByText("GITLAB$WEBHOOK_MANAGER_TITLE"),
      ).not.toBeInTheDocument();
    });
  });
});
