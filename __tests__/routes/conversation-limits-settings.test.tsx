import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import ConversationLimitsSettingsScreen from "#/routes/conversation-limits-settings";
import type { SettingsValue } from "#/types/settings";

function renderScreen(conversationSettings: Record<string, SettingsValue>) {
  vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
    ...MOCK_DEFAULT_USER_SETTINGS,
    conversation_settings: conversationSettings,
  });
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(<ConversationLimitsSettingsScreen />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("ConversationLimitsSettingsScreen", () => {
  it("shows the conversation limits and none of the runtime fields", async () => {
    renderScreen({ max_iterations: 500, default_runtime: "docker" });

    await screen.findByTestId("conversation-limits-settings-screen");

    expect(screen.getByTestId("sdk-settings-max_iterations")).toBeVisible();
    expect(screen.getByTestId("sdk-settings-max_budget_per_run")).toBeVisible();
    expect(
      screen.queryByTestId("sdk-settings-default_runtime"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("sdk-settings-container_memory"),
    ).not.toBeInTheDocument();
  });

  it("labels the budget as the default for new conversations", async () => {
    renderScreen({ max_iterations: 500 });

    await screen.findByTestId("conversation-limits-settings-screen");

    expect(
      screen.getByText("SCHEMA$MAX_BUDGET_PER_RUN$LABEL"),
    ).toBeInTheDocument();
  });

  it("saves the budget as a conversation_settings_diff", async () => {
    const user = userEvent.setup();
    const saveSettingsSpy = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);
    renderScreen({ max_iterations: 500 });

    await screen.findByTestId("conversation-limits-settings-screen");
    await user.type(
      screen.getByTestId("sdk-settings-max_budget_per_run"),
      "2.5",
    );
    await user.click(screen.getByTestId("save-button"));

    expect(saveSettingsSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        conversation_settings_diff: { max_budget_per_run: 2.5 },
      }),
    );
  });
});
