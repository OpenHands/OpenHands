import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SetupLlmStep } from "#/components/features/onboarding/steps/setup-llm-step";
import type { SdkSectionSaveControl } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { renderWithProviders } from "../../../test-utils";

const saveProfile = vi.hoisted(() => vi.fn());
const activateProfile = vi.hoisted(() => vi.fn());
const applyAgentProfile = vi.hoisted(() => vi.fn());
const getSavedSettings = vi.hoisted(() => vi.fn());
const displayErrorToast = vi.hoisted(() => vi.fn());
let finishSettingsSave: (() => void) | undefined;
const formState = vi.hoisted(() => ({
  llm: {} as Record<string, unknown>,
  dirtyLlm: undefined as Record<string, unknown> | undefined,
  backendKind: "local",
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({ backend: { kind: formState.backendKind } }),
}));

vi.mock("#/api/profiles-service/profiles-service.api", () => ({
  default: { saveProfile, activateProfile },
}));

vi.mock("#/api/settings-service/settings-service.api", () => ({
  default: {
    fetchSettingsFromApi: getSavedSettings,
    invalidateCache: vi.fn(),
  },
}));

vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME: "default",
  default: {
    saveProfile: applyAgentProfile,
    getProfile: vi
      .fn()
      .mockResolvedValue({ profile: { id: "onboarding-agent" } }),
    activateProfile: vi.fn().mockResolvedValue(undefined),
  },
}));

vi.mock("#/hooks/query/use-free-models", () => ({
  useDefaultModel: () => null,
  useDefaultModelReady: () => true,
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast,
}));

vi.mock("#/routes/llm-settings", async () => {
  const React = await import("react");
  return {
    LlmSettingsScreen: ({
      onSaveControlChange,
      onSaveSuccess,
    }: {
      onSaveControlChange: (control: SdkSectionSaveControl) => void;
      onSaveSuccess: () => void;
    }) => {
      const canonicalLlm = formState.llm;
      finishSettingsSave = onSaveSuccess;
      React.useEffect(() => {
        const buildControl = (isDirty: boolean): SdkSectionSaveControl => ({
          save: () => {
            onSaveControlChange(buildControl(false));
            finishSettingsSave?.();
          },
          isSaving: false,
          isDirty,
          values: Object.fromEntries(
            Object.entries(canonicalLlm).map(([key, value]) => [
              `llm.${key}`,
              String(value),
            ]),
          ),
          view: "basic",
          getDirtyPayload: () => ({ llm: formState.dirtyLlm ?? canonicalLlm }),
          getSavePayload: () => ({
            agent_settings_diff: { llm: formState.dirtyLlm ?? canonicalLlm },
          }),
        });
        onSaveControlChange(buildControl(true));
      }, [onSaveControlChange]);
      return <div data-testid="llm-settings-screen" />;
    },
  };
});

describe("SetupLlmStep", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    formState.backendKind = "local";
    formState.dirtyLlm = undefined;
    formState.llm = {
      model: "gpt-5.6-luna",
      auth_type: "subscription",
      subscription_vendor: "openai",
      temperature: 0.2,
    };
    finishSettingsSave = undefined;
    saveProfile.mockResolvedValue(undefined);
    activateProfile.mockResolvedValue(undefined);
    applyAgentProfile.mockResolvedValue(undefined);
    getSavedSettings.mockImplementation(async () => ({
      agent_settings: { llm: formState.llm },
    }));
  });

  it("persists and activates the canonically serialized subscription profile", async () => {
    const onNext = vi.fn();
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={onNext} />);

    await userEvent.click(await screen.findByTestId("onboarding-llm-next"));

    await waitFor(() => {
      expect(saveProfile).toHaveBeenCalledWith("gpt-5.6-luna", {
        llm: {
          model: "gpt-5.6-luna",
          auth_type: "subscription",
          subscription_vendor: "openai",
          temperature: 0.2,
        },
        include_secrets: true,
      });
    });
    expect(activateProfile).toHaveBeenCalledWith("gpt-5.6-luna");
    expect(applyAgentProfile).toHaveBeenCalledWith("default", {
      agent_kind: "openhands",
      llm_profile_ref: "gpt-5.6-luna",
    });
    expect(onNext).toHaveBeenCalledTimes(1);
  });

  it("preserves typed advanced options when creating the activated profile", async () => {
    formState.llm = {
      model: "openai/gpt-4o-mini",
      api_key: "test-key",
      base_url: "http://localhost:19118/v1",
      timeout: 3,
      num_retries: 1,
      retry_min_wait: 0,
      retry_max_wait: 0,
      stream: true,
    };
    const onNext = vi.fn();
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={onNext} />);

    await userEvent.click(await screen.findByTestId("onboarding-llm-next"));

    await waitFor(() => {
      expect(saveProfile).toHaveBeenCalledWith("gpt-4o-mini", {
        llm: formState.llm,
        include_secrets: true,
      });
    });
    expect(activateProfile).toHaveBeenCalledWith("gpt-4o-mini");
    expect(applyAgentProfile).toHaveBeenCalledWith("default", {
      agent_kind: "openhands",
      llm_profile_ref: "gpt-4o-mini",
    });
    expect(onNext).toHaveBeenCalledTimes(1);
  });

  it("preserves unchanged endpoint, credentials and options outside the settings diff", async () => {
    formState.llm = {
      model: "openai/mock-onboarding-model",
      base_url: "http://localhost:19118/v1",
      api_key: "encrypted:test-key",
      temperature: 0.2,
      timeout: 30,
    };
    formState.dirtyLlm = { model: formState.llm.model };
    const onNext = vi.fn();
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={onNext} />);

    await userEvent.click(await screen.findByTestId("onboarding-llm-next"));

    await waitFor(() => expect(onNext).toHaveBeenCalledTimes(1));
    expect(getSavedSettings).toHaveBeenCalledWith("encrypted");
    expect(saveProfile).toHaveBeenCalledWith("mock-onboarding-model", {
      llm: formState.llm,
      include_secrets: true,
    });
  });

  it("keeps Cloud onboarding on its settings save path", async () => {
    formState.backendKind = "cloud";
    const onNext = vi.fn();
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={onNext} />);

    await userEvent.click(await screen.findByTestId("onboarding-llm-next"));

    await waitFor(() => expect(onNext).toHaveBeenCalledTimes(1));
    expect(getSavedSettings).not.toHaveBeenCalled();
    expect(saveProfile).not.toHaveBeenCalled();
    expect(activateProfile).not.toHaveBeenCalled();
    expect(applyAgentProfile).not.toHaveBeenCalled();
  });

  it.each([
    ["saved settings read", getSavedSettings],
    ["profile save", saveProfile],
    ["profile activation", activateProfile],
  ])("does not advance when %s fails", async (_label, failingMutation) => {
    failingMutation.mockRejectedValueOnce(new Error("profile unavailable"));
    const onNext = vi.fn();
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={onNext} />);

    await userEvent.click(await screen.findByTestId("onboarding-llm-next"));

    await waitFor(() => expect(displayErrorToast).toHaveBeenCalled());
    expect(onNext).not.toHaveBeenCalled();
    expect(applyAgentProfile).not.toHaveBeenCalled();
    const nextButton = screen.getByTestId("onboarding-llm-next");
    expect(nextButton).toBeEnabled();

    await userEvent.click(nextButton);
    await waitFor(() => expect(onNext).toHaveBeenCalledTimes(1));
  });
});
