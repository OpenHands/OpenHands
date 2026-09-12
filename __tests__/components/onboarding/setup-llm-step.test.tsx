import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SetupLlmStep } from "#/components/features/onboarding/steps/setup-llm-step";
import type { SdkSectionSaveControl } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { renderWithProviders } from "../../../test-utils";

const saveProfile = vi.hoisted(() => vi.fn());
const activateProfile = vi.hoisted(() => vi.fn());
const applyAgentProfile = vi.hoisted(() => vi.fn());
const displayErrorToast = vi.hoisted(() => vi.fn());
let finishSettingsSave: (() => void) | undefined;

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({ backend: { kind: "local" } }),
}));

vi.mock("#/hooks/mutation/use-save-llm-profile", () => ({
  useSaveLlmProfile: () => ({ mutateAsync: saveProfile }),
}));

vi.mock("#/hooks/mutation/use-activate-llm-profile", () => ({
  useActivateLlmProfile: () => ({ mutateAsync: activateProfile }),
}));

vi.mock("#/hooks/mutation/use-apply-onboarding-agent-profile", () => ({
  useApplyOnboardingAgentProfile: () => applyAgentProfile,
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
  const canonicalLlm = {
    model: "gpt-5.6-luna",
    auth_type: "subscription",
    subscription_vendor: "openai",
    temperature: 0.2,
  };

  return {
    LlmSettingsScreen: ({
      onSaveControlChange,
      onSaveSuccess,
    }: {
      onSaveControlChange: (control: SdkSectionSaveControl) => void;
      onSaveSuccess: () => void;
    }) => {
      finishSettingsSave = onSaveSuccess;
      React.useEffect(() => {
        const buildControl = (isDirty: boolean): SdkSectionSaveControl => ({
          save: () => {
            onSaveControlChange(buildControl(false));
            finishSettingsSave?.();
          },
          isSaving: false,
          isDirty,
          values: {
            "llm.model": canonicalLlm.model,
            "llm.auth_type": canonicalLlm.auth_type,
            "llm.subscription_vendor": canonicalLlm.subscription_vendor,
            "llm.temperature": String(canonicalLlm.temperature),
          },
          view: "basic",
          getDirtyPayload: () => ({ llm: canonicalLlm }),
          getSavePayload: () => ({
            agent_settings_diff: { llm: canonicalLlm },
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
    finishSettingsSave = undefined;
    saveProfile.mockResolvedValue(undefined);
    activateProfile.mockResolvedValue(undefined);
    applyAgentProfile.mockResolvedValue(undefined);
  });

  it("persists and activates the canonically serialized subscription profile", async () => {
    const onNext = vi.fn();
    renderWithProviders(<SetupLlmStep onBack={vi.fn()} onNext={onNext} />);

    await userEvent.click(await screen.findByTestId("onboarding-llm-next"));

    await waitFor(() => {
      expect(saveProfile).toHaveBeenCalledWith({
        name: "gpt-5.6-luna",
        request: {
          llm: {
            model: "gpt-5.6-luna",
            auth_type: "subscription",
            subscription_vendor: "openai",
            temperature: 0.2,
          },
          include_secrets: true,
        },
      });
    });
    expect(activateProfile).toHaveBeenCalledWith("gpt-5.6-luna");
    expect(applyAgentProfile).toHaveBeenCalledWith({
      agent_kind: "openhands",
      llm_profile_ref: "gpt-5.6-luna",
    });
    expect(onNext).toHaveBeenCalledTimes(1);
  });

  it.each([
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
