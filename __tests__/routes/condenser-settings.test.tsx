import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import CondenserSettingsScreen from "#/routes/condenser-settings";

const ACTIVE_PROFILE = {
  id: "profile-default",
  name: "default",
  revision: 1,
  agent_kind: "openhands",
  llm_profile_ref: "default",
  mcp_server_refs: null,
  condenser: { enable_default_condenser: false },
} as const;

function renderCondenserSettingsScreen() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(<CondenserSettingsScreen />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
}

describe("CondenserSettingsScreen", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
      ...MOCK_DEFAULT_USER_SETTINGS,
      default_profile_migrated: true,
      agent_settings: {
        ...MOCK_DEFAULT_USER_SETTINGS.agent_settings,
        condenser: { enable_default_condenser: true },
      },
    });
    vi.spyOn(AgentProfilesService, "listProfiles").mockResolvedValue({
      profiles: [ACTIVE_PROFILE],
      active_agent_profile_id: ACTIVE_PROFILE.id,
    } as never);
    vi.spyOn(AgentProfilesService, "getProfile").mockResolvedValue({
      name: ACTIVE_PROFILE.name,
      profile: ACTIVE_PROFILE,
    } as never);
  });

  it("edits the active profile's condenser, not the global one", async () => {
    const saveSettings = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);
    const saveProfile = vi
      .spyOn(AgentProfilesService, "saveProfile")
      .mockResolvedValue({} as never);

    renderCondenserSettingsScreen();

    const toggle = await screen.findByTestId(
      "sdk-settings-condenser.enable_default_condenser",
    );
    expect(toggle).not.toBeChecked();
    await userEvent.click(toggle.closest("label")!);
    await userEvent.click(screen.getByTestId("save-button"));

    await waitFor(() => expect(saveProfile).toHaveBeenCalledTimes(1));
    expect(saveProfile).toHaveBeenCalledWith(
      "default",
      expect.objectContaining({
        condenser: expect.objectContaining({ enable_default_condenser: true }),
      }),
    );
    expect(saveSettings).not.toHaveBeenCalled();
  });
});
