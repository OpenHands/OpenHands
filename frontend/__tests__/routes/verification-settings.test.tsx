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
});

describe("VerificationSettingsScreen", () => {
  it("keeps confirmation mode toggle visible in the header", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(buildSettings());

    renderVerificationSettingsScreen();

    await screen.findByTestId("verification-settings-screen");

    expect(
      screen.getByTestId("confirmation-mode-toggle"),
    ).toBeInTheDocument();
    // Security analyzer is hidden when confirmation mode is off
    expect(
      screen.queryByTestId("security-analyzer-input"),
    ).not.toBeInTheDocument();
  });

  it("shows the security analyzer when confirmation mode is enabled", async () => {
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      buildSettings({
        conversation_settings: {
          ...MOCK_DEFAULT_USER_SETTINGS.conversation_settings,
          confirmation_mode: true,
          security_analyzer: "llm",
        },
      }),
    );

    renderVerificationSettingsScreen();

    await screen.findByTestId("verification-settings-screen");

    // Security analyzer should appear because confirmation mode is on
    expect(
      screen.getByTestId("security-analyzer-input"),
    ).toBeInTheDocument();
  });
});

describe("clientLoader permission checks", () => {
  it("should export a clientLoader for route protection", () => {
    expect(clientLoader).toBeDefined();
    expect(typeof clientLoader).toBe("function");
  });
});
