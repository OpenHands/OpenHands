import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import VerificationSettingsScreen, {
  clientLoader,
} from "#/routes/verification-settings";
import { Settings } from "#/types/settings";

const mockUseConfig = vi.fn();

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => mockUseConfig(),
}));

vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => ({ data: true }),
}));

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

beforeEach(() => {
  vi.restoreAllMocks();
  mockUseConfig.mockReturnValue({
    data: { app_mode: "oss" },
    isLoading: false,
  });
});

describe("VerificationSettingsScreen", () => {
  it("renders critic controls alongside existing confirmation settings", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          verification: {
            critic_enabled: true,
            enable_iterative_refinement: false,
          },
        },
      }),
    );

    renderVerificationSettingsScreen();

    await screen.findByTestId("verification-settings-screen");

    expect(
      screen.getByTestId("sdk-settings-verification.critic_enabled"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId(
        "sdk-settings-verification.enable_iterative_refinement",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("sdk-settings-confirmation_mode"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("sdk-settings-verification.critic_mode"),
    ).not.toBeInTheDocument();
    // Security analyzer is hidden when confirmation mode is off
    expect(
      screen.queryByTestId("sdk-settings-security_analyzer"),
    ).not.toBeInTheDocument();
  });

  it("keeps security analyzer in the existing advanced view", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        conversation_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.conversation_settings,
          confirmation_mode: true,
          security_analyzer: "llm",
        },
        agent_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
          verification: {
            critic_enabled: true,
            enable_iterative_refinement: false,
          },
        },
      }),
    );

    renderVerificationSettingsScreen();

    await screen.findByTestId("verification-settings-screen");

    expect(
      screen.queryByTestId("sdk-settings-security_analyzer"),
    ).not.toBeInTheDocument();

    await userEvent.click(screen.getByTestId("sdk-section-advanced-toggle"));

    expect(
      await screen.findByTestId("sdk-settings-security_analyzer"),
    ).toBeInTheDocument();
    expect(
      await screen.findByTestId("sdk-settings-verification.critic_mode"),
    ).toBeInTheDocument();
  });
});

describe("clientLoader permission checks", () => {
  it("should export a clientLoader for route protection", () => {
    expect(clientLoader).toBeDefined();
    expect(typeof clientLoader).toBe("function");
  });
});
