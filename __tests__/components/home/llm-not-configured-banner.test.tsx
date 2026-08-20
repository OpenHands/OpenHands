import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { LlmNotConfiguredBanner } from "#/components/features/home/llm-not-configured-banner";
import { NavigationProvider } from "#/context/navigation-context";
import { useSettings } from "#/hooks/query/use-settings";
import { useConfig } from "#/hooks/query/use-config";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import SettingsService from "#/api/settings-service/settings-service.api";
import OptionService from "#/api/option-service/option-service.api";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { Settings } from "#/types/settings";
import { WebClientConfig } from "#/api/option-service/option.types";

function buildSettings(overrides: Partial<Settings> = {}): Settings {
  return {
    ...MOCK_DEFAULT_USER_SETTINGS,
    ...overrides,
    agent_settings:
      overrides.agent_settings ?? MOCK_DEFAULT_USER_SETTINGS.agent_settings,
  };
}

function buildConfig(hideLlmSettings = false): WebClientConfig {
  return {
    feature_flags: {
      hide_llm_settings: hideLlmSettings,
      hide_users_page: true,
    },
    providers_configured: [],
    maintenance_start_time: null,
    recaptcha_site_key: null,
    faulty_models: [],
    error_message: null,
    updated_at: "2026-01-01T00:00:00.000Z",
  };
}

// Renders the banner next to a probe that flips on once both underlying
// queries have resolved. The "hidden" assertions wait on this marker so they
// observe the settled state rather than the (always-empty) loading frame.
function ProbedBanner() {
  const settings = useSettings();
  const config = useConfig();
  const profiles = useLlmProfiles();
  return (
    <>
      <LlmNotConfiguredBanner />
      {settings.isFetched && config.isFetched && profiles.isFetched ? (
        <div data-testid="queries-settled" />
      ) : null}
    </>
  );
}

function renderBanner() {
  const navigate = vi.fn();
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <NavigationProvider
        value={{
          currentPath: "/",
          conversationId: null,
          isNavigating: false,
          navigate,
        }}
      >
        <ProbedBanner />
      </NavigationProvider>
    </QueryClientProvider>,
  );
  return { navigate };
}

describe("LlmNotConfiguredBanner", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(OptionService, "getConfig").mockResolvedValue(buildConfig());
    vi.spyOn(ProfilesService, "listProfiles").mockResolvedValue({
      profiles: [],
      active_profile: null,
    });
  });

  it("warns the user when no LLM API key is set (the skip-onboarding case)", async () => {
    // Arrange: an OpenHands agent with no key — what "Skip for now" leaves.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ llm_api_key_set: false }),
    );

    // Act
    renderBanner();

    // Assert
    expect(
      await screen.findByTestId("home-llm-not-configured-banner"),
    ).toBeInTheDocument();
  });

  it("warns when only a stale settings key is set but no active profile (local)", async () => {
    // Arrange: the delete-all state — agent_settings still holds a key, but no
    // profile backs it. In local mode profiles are the source of truth, so a
    // lingering settings key must not count as configured.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ llm_api_key_set: true }),
    );

    // Act
    renderBanner();

    // Assert
    expect(
      await screen.findByTestId("home-llm-not-configured-banner"),
    ).toBeInTheDocument();
  });

  it("stays hidden once an active LLM profile has a saved API key", async () => {
    // Arrange: the profile endpoint is the source of truth for the active
    // profile shown in LLM settings, even if the legacy settings flag is stale.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ llm_api_key_set: false }),
    );
    vi.spyOn(ProfilesService, "listProfiles").mockResolvedValue({
      profiles: [
        {
          name: "active-profile",
          model: "openai/gpt-4.1",
          base_url: null,
          api_key_set: true,
        },
      ],
      active_profile: "active-profile",
    });

    // Act
    renderBanner();
    await screen.findByTestId("queries-settled");

    // Assert
    expect(
      screen.queryByTestId("home-llm-not-configured-banner"),
    ).not.toBeInTheDocument();
  });

  it("stays hidden for ACP agents, which own their LLM and need no key", async () => {
    // Arrange: ACP agent, no key — must not be nagged.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        llm_api_key_set: false,
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          agent_kind: "acp",
        },
      }),
    );

    // Act
    renderBanner();
    await screen.findByTestId("queries-settled");

    // Assert
    expect(
      screen.queryByTestId("home-llm-not-configured-banner"),
    ).not.toBeInTheDocument();
  });

  it("stays hidden when the LLM settings page is hidden by feature flag", async () => {
    // Arrange: no key, but the settings page is hidden — routing the user to
    // /settings/llm would dead-end, so the hook treats the LLM as configured.
    vi.spyOn(OptionService, "getConfig").mockResolvedValue(buildConfig(true));
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ llm_api_key_set: false }),
    );

    // Act
    renderBanner();
    await screen.findByTestId("queries-settled");

    // Assert
    expect(
      screen.queryByTestId("home-llm-not-configured-banner"),
    ).not.toBeInTheDocument();
  });

  it("warns when settings 404 — the genuine new-user state — despite the error flag", async () => {
    // Arrange: a 404 means no settings exist yet. useSettings maps it to
    // DEFAULT_SETTINGS (no key) while keeping isError set, so the banner must
    // still appear — this is the real "Skip for now" state. Guards against
    // conflating this with a transient fetch error that should suppress it.
    vi.spyOn(SettingsService, "getSettings").mockRejectedValue({ status: 404 });

    // Act
    renderBanner();

    // Assert
    expect(
      await screen.findByTestId("home-llm-not-configured-banner"),
    ).toBeInTheDocument();
  });

  it("routes the user to LLM settings when the action is clicked", async () => {
    // Arrange
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({ llm_api_key_set: false }),
    );
    const user = userEvent.setup();
    const { navigate } = renderBanner();

    // Act
    await user.click(
      await screen.findByTestId("home-llm-not-configured-action"),
    );

    // Assert
    expect(navigate).toHaveBeenCalledWith("/settings/llm");
  });
});
