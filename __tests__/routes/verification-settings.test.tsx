import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import VerificationSettingsScreen from "#/routes/verification-settings";
import { Settings } from "#/types/settings";

function buildSettings(overrides: Partial<Settings> = {}): Settings {
  return {
    ...MOCK_DEFAULT_USER_SETTINGS,
    ...overrides,
    agent_settings: {
      ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
      ...overrides.agent_settings,
    },
    conversation_settings: {
      ...MOCK_DEFAULT_USER_SETTINGS.conversation_settings,
      ...overrides.conversation_settings,
    },
    agent_settings_schema:
      overrides.agent_settings_schema ??
      MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema,
    conversation_settings_schema:
      overrides.conversation_settings_schema ??
      MOCK_DEFAULT_USER_SETTINGS.conversation_settings_schema,
  };
}

function renderVerificationSettingsScreen() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return render(<VerificationSettingsScreen />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
}

// Covered by its own tests; saves here only need it settled.
vi.mock("#/hooks/use-migrate-default-agent-profile", () => ({
  ensureDefaultAgentProfileMigrated: vi.fn().mockResolvedValue(true),
}));

const ACTIVE_PROFILE = {
  id: "profile-default",
  name: "default",
  revision: 1,
  agent_kind: "openhands",
  llm_profile_ref: "default",
  mcp_server_refs: null,
} as const;

/** The critic settings the active agent profile stores. */
function withActiveProfileVerification(verification: Record<string, unknown>) {
  vi.spyOn(AgentProfilesService, "listProfiles").mockResolvedValue({
    profiles: [ACTIVE_PROFILE],
    active_agent_profile_id: ACTIVE_PROFILE.id,
  } as never);
  vi.spyOn(AgentProfilesService, "getProfile").mockResolvedValue({
    name: ACTIVE_PROFILE.name,
    profile: { ...ACTIVE_PROFILE, verification },
  } as never);
}

beforeEach(() => {
  vi.restoreAllMocks();
  withActiveProfileVerification({
    critic_enabled: false,
    enable_iterative_refinement: false,
  });
});

describe("VerificationSettingsScreen", () => {
  it("renders the active profile's critic controls in basic view", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    withActiveProfileVerification({
      critic_enabled: true,
      enable_iterative_refinement: false,
    });

    renderVerificationSettingsScreen();

    await screen.findByTestId("verification-settings-screen");

    // Critical-prominence fields are visible in the basic view
    expect(
      screen.getByTestId("sdk-settings-verification.critic_enabled"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId(
        "sdk-settings-verification.enable_iterative_refinement",
      ),
    ).toBeInTheDocument();

    // The critic LLM API key is optional because the SDK falls back to the
    // active LLM API key when this field is blank. It is still surfaced
    // alongside the other critical-prominence fields when the critic is on.
    const apiKeyInput = screen.getByTestId(
      "sdk-settings-verification.critic_api_key",
    );
    expect(apiKeyInput).toBeInTheDocument();
    expect(apiKeyInput).toHaveAttribute("type", "password");
    expect(apiKeyInput).not.toBeRequired();

    // The accompanying help link points users at OpenHands Cloud, mirroring
    // the hint we already show under the LLM provider's API key field.
    const helpLink = screen.getByTestId(
      "help-link-verification.critic_api_key",
    );
    expect(helpLink).toBeInTheDocument();
    expect(
      helpLink.querySelector(
        'a[href="https://app.all-hands.dev/settings/api-keys"]',
      ),
    ).not.toBeNull();

    // Major-prominence fields (confirmation_mode) are hidden in basic view
    expect(
      screen.queryByTestId("sdk-settings-confirmation_mode"),
    ).not.toBeInTheDocument();
  });

  it("hides the critic API key field when the profile's critic is disabled", async () => {
    // Global settings still say "on"; the profile decides.
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          verification: { critic_enabled: true },
        },
      }),
    );

    renderVerificationSettingsScreen();

    await screen.findByTestId("sdk-settings-verification.critic_enabled");
    expect(
      screen.queryByTestId("sdk-settings-verification.critic_api_key"),
    ).not.toBeInTheDocument();
  });

  it("saves the critic behavior to the active profile and its API key globally", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());
    const saveSettings = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);
    const saveProfile = vi
      .spyOn(AgentProfilesService, "saveProfile")
      .mockResolvedValue({} as never);

    renderVerificationSettingsScreen();

    const critic = await screen.findByTestId(
      "sdk-settings-verification.critic_enabled",
    );
    await userEvent.click(critic.closest("label")!);
    await userEvent.type(
      await screen.findByTestId("sdk-settings-verification.critic_api_key"),
      "sk-critic",
    );
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => expect(saveProfile).toHaveBeenCalledTimes(1));
    const [name, saved] = saveProfile.mock.calls[0];
    expect(name).toBe("default");
    expect(saved).toMatchObject({
      agent_kind: "openhands",
      llm_profile_ref: "default",
      verification: { critic_enabled: true },
    });
    expect(
      (saved as unknown as { verification: Record<string, unknown> })
        .verification,
    ).not.toHaveProperty("critic_api_key");
    expect(saveSettings).toHaveBeenCalledWith(
      expect.objectContaining({
        agent_settings_diff: { verification: { critic_api_key: "sk-critic" } },
      }),
    );
  });
  it("shows confirmation controls in the advanced view", async () => {
    // Set confirmation_mode to true so inferInitialView picks "advanced"
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        conversation_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.conversation_settings,
          confirmation_mode: true,
        },
      }),
    );

    renderVerificationSettingsScreen();

    await screen.findByTestId("verification-settings-screen");

    // Confirmation mode (major prominence) should be visible
    expect(
      screen.getByTestId("sdk-settings-confirmation_mode"),
    ).toBeInTheDocument();
    // Security analyzer depends on confirmation_mode being true
    expect(
      screen.getByTestId("sdk-settings-security_analyzer"),
    ).toBeInTheDocument();
  });

  it("deduplicates legacy agent verification fields in favor of conversation settings", async () => {
    const legacyAgentSchema = structuredClone(
      MOCK_DEFAULT_USER_SETTINGS.agent_settings_schema!,
    );
    const verificationSection = legacyAgentSchema.sections.find(
      (section) => section.key === "verification",
    );
    expect(verificationSection).toBeDefined();
    verificationSection!.fields.push(
      {
        key: "verification.confirmation_mode",
        label: "Legacy confirmation mode",
        description: "Legacy agent-owned confirmation mode.",
        section: "verification",
        section_label: "Verification",
        value_type: "boolean",
        default: false,
        choices: [],
        depends_on: [],
        prominence: "major",
        secret: false,
        required: false,
      },
      {
        key: "verification.security_analyzer",
        label: "Legacy security analyzer",
        description: "Legacy agent-owned security analyzer.",
        section: "verification",
        section_label: "Verification",
        value_type: "string",
        default: "llm",
        choices: [
          { label: "llm", value: "llm" },
          { label: "none", value: "none" },
        ],
        depends_on: ["verification.confirmation_mode"],
        prominence: "major",
        secret: false,
        required: false,
      },
    );

    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings_schema: legacyAgentSchema,
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          verification: {
            critic_enabled: false,
            enable_iterative_refinement: false,
            confirmation_mode: true,
            security_analyzer: "llm",
          },
        },
        conversation_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.conversation_settings,
          confirmation_mode: true,
          security_analyzer: "llm",
        },
      }),
    );

    renderVerificationSettingsScreen();

    await screen.findByTestId("verification-settings-screen");

    expect(
      screen.getByTestId("sdk-settings-confirmation_mode"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("sdk-settings-security_analyzer"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("sdk-settings-verification.confirmation_mode"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("sdk-settings-verification.security_analyzer"),
    ).not.toBeInTheDocument();
  });
});
