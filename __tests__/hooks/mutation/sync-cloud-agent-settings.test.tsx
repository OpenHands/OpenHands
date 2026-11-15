import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import type { AgentProfile } from "#/api/agent-profiles-service/agent-profiles-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import SettingsService from "#/api/settings-service/settings-service.api";
import { useActivateAgentProfile } from "#/hooks/mutation/use-activate-agent-profile";
import { useSaveAgentProfile } from "#/hooks/mutation/use-save-agent-profile";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  default: {
    activateProfile: vi.fn(),
    listProfiles: vi.fn(),
    getProfile: vi.fn(),
    saveProfile: vi.fn(),
  },
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
  displaySuccessToast: vi.fn(),
}));

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-key",
  kind: "cloud",
};

const localBackend: Backend = {
  id: "local-1",
  name: "Local",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};

const openHandsProfile = {
  agent_kind: "openhands",
  llm_profile_ref: "claude",
} as AgentProfile;

const codexProfile = {
  agent_kind: "acp",
  acp_server: "custom",
  acp_command: "npx -y my-acp --stdio",
  acp_args: ["--verbose"],
  acp_model: "gpt-5.5",
} as AgentProfile;

function activate(backend: Backend) {
  setRegisteredBackends([backend]);
  setActiveSelection({ backendId: backend.id });
}

/** Stores the profiles and makes `activeId` the active one. */
function serveProfiles(activeId: string, stored: Record<string, AgentProfile>) {
  vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
    active_agent_profile_id: activeId,
    profiles: Object.entries(stored).map(([name, profile]) => ({
      id: `${name}-id`,
      name,
      agent_kind: profile.agent_kind,
      llm_profile_ref: null,
    })),
  } as never);
  vi.mocked(AgentProfilesService.getProfile).mockImplementation(
    async (name: string) => ({ name, profile: stored[name] }),
  );
}

function renderWithClient<T>(hook: () => T) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return renderHook(hook, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
}

describe("Cloud agent_settings sync with the active agent profile", () => {
  let saveSettings: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    saveSettings = vi
      .spyOn(SettingsService, "saveSettings")
      .mockResolvedValue(true);
    vi.mocked(AgentProfilesService.activateProfile).mockResolvedValue({
      id: "openhands-id",
      message: "activated",
      agent_settings_applied: false,
    });
    vi.mocked(AgentProfilesService.saveProfile).mockResolvedValue({
      name: "codex",
      message: "saved",
    });
  });

  afterEach(() => {
    __resetActiveStoreForTests();
    vi.clearAllMocks();
  });

  it("records the OpenHands kind when an OpenHands profile is activated on Cloud", async () => {
    activate(cloudBackend);
    serveProfiles("openhands-id", { openhands: openHandsProfile });
    const { result } = renderWithClient(() => useActivateAgentProfile());

    await result.current.mutateAsync("openhands-id");

    expect(saveSettings).toHaveBeenCalledExactlyOnceWith({
      agent_settings_diff: { agent_kind: "openhands" },
    });
  });

  it("records the profile's ACP block when an ACP profile is activated on Cloud", async () => {
    activate(cloudBackend);
    serveProfiles("codex-id", { codex: codexProfile });
    const { result } = renderWithClient(() => useActivateAgentProfile());

    await result.current.mutateAsync("codex-id");

    expect(saveSettings).toHaveBeenCalledExactlyOnceWith({
      agent_settings_diff: {
        agent_kind: "acp",
        acp_server: "custom",
        acp_command: ["npx", "-y", "my-acp", "--stdio"],
        acp_args: ["--verbose"],
        acp_model: "gpt-5.5",
      },
    });
  });

  it("keeps the activation and reports the error when the settings sync fails", async () => {
    activate(cloudBackend);
    serveProfiles("openhands-id", { openhands: openHandsProfile });
    saveSettings.mockRejectedValue(new Error("Settings unavailable"));
    const { result } = renderWithClient(() => useActivateAgentProfile());

    result.current.mutate("openhands-id");

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(displayErrorToast).toHaveBeenCalledOnce();
  });

  it("does not write agent_settings when the server already applied them", async () => {
    activate(cloudBackend);
    serveProfiles("openhands-id", { openhands: openHandsProfile });
    vi.mocked(AgentProfilesService.activateProfile).mockResolvedValue({
      id: "openhands-id",
      message: "activated",
      agent_settings_applied: true,
    });
    const { result } = renderWithClient(() => useActivateAgentProfile());

    await result.current.mutateAsync("openhands-id");

    expect(saveSettings).not.toHaveBeenCalled();
  });

  it("records the new kind when the active profile is saved on Cloud, and ignores other profiles", async () => {
    activate(cloudBackend);
    serveProfiles("codex-id", { codex: codexProfile, other: openHandsProfile });
    const { result } = renderWithClient(() => useSaveAgentProfile());

    await result.current.mutateAsync({
      name: "other",
      profile: { agent_kind: "openhands" },
    });
    expect(saveSettings).not.toHaveBeenCalled();

    await result.current.mutateAsync({
      name: "codex",
      profile: { agent_kind: "acp" },
    });
    expect(saveSettings).toHaveBeenCalledExactlyOnceWith({
      agent_settings_diff: expect.objectContaining({
        agent_kind: "acp",
        acp_server: "custom",
      }),
    });
  });

  it("never writes agent_settings on a local backend", async () => {
    activate(localBackend);
    serveProfiles("codex-id", { codex: codexProfile });
    const activation = renderWithClient(() => useActivateAgentProfile());
    const save = renderWithClient(() => useSaveAgentProfile());

    await activation.result.current.mutateAsync("codex-id");
    await save.result.current.mutateAsync({
      name: "codex",
      profile: { agent_kind: "acp" },
    });

    expect(saveSettings).not.toHaveBeenCalled();
    expect(AgentProfilesService.listProfiles).not.toHaveBeenCalled();
  });
});
