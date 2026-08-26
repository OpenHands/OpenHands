import React from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { LlmSettingsScreen } from "#/routes/llm-settings";
import type { SdkSectionSaveControl } from "#/components/features/settings/sdk-settings/sdk-section-page";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useSaveLlmProfile } from "#/hooks/mutation/use-save-llm-profile";
import { useActivateLlmProfile } from "#/hooks/mutation/use-activate-llm-profile";
import { useApplyOnboardingAgentProfile } from "#/hooks/mutation/use-apply-onboarding-agent-profile";
import { deriveProfileNameFromModel } from "#/utils/derive-profile-name";

interface SetupLlmStepProps {
  onBack: () => void;
  onNext: () => void;
}

/**
 * Pre-fills the LLM form with the OpenAI GPT-5.5 model. The SDK's
 * bare default model is `gpt-5.5`; Canvas stores provider-qualified
 * LiteLLM model ids, so the onboarding override uses this OpenAI-prefixed
 * model id. Keeping this as an explicit override marks the model dirty so
 * the Next button persists the suggested default immediately.
 */
export const ONBOARDING_DEFAULT_LLM_MODEL = "openai/gpt-5.5";

/**
 * Step 2: embed the LLM settings form. The screen runs in `embedded`
 * mode (so it doesn't render its own sticky Save bar) and with
 * `hideSaveButton` set, surfacing its save state via
 * `onSaveControlChange`. We then render a single Next button at the
 * modal footer level matching the other onboarding steps; clicking
 * Next saves the form and `onSaveSuccess` advances to the next step.
 *
 * If the form happens to be untouched (no dirty fields), Next falls
 * through to advancing without a save call, so users with already-
 * configured settings aren't blocked.
 *
 * Note: returning Cloud users who already have an LLM configured are
 * intercepted upstream by `OnboardingHost`, so they never reach this
 * step. Users who do reach it are first-time installs (Cloud or Local)
 * who want the OpenAI/GPT-5.5 default pre-filled.
 */
export function SetupLlmStep({ onBack, onNext }: SetupLlmStepProps) {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const isLocalBackend = backend.kind === "local";
  const saveProfile = useSaveLlmProfile();
  const activateProfile = useActivateLlmProfile();
  const applyAgentProfile = useApplyOnboardingAgentProfile();
  const [saveControl, setSaveControl] =
    React.useState<SdkSectionSaveControl | null>(null);
  const [isFinalizing, setIsFinalizing] = React.useState(false);

  // On local backends the LLM profiles list is the user-facing source of
  // truth; without this step the form save only updates agent_settings and
  // the new config never shows up in the profiles list ("ghost profile").
  // Returns the saved LLM profile name so the caller can point the active
  // AGENT profile at it (conversations launch from the agent profile, not the
  // active LLM profile).
  const persistAsProfile = React.useCallback(async (): Promise<
    string | null
  > => {
    if (!isLocalBackend || !saveControl) return null;
    const values = saveControl.values;
    const model =
      typeof values["llm.model"] === "string" ? values["llm.model"] : "";
    if (!model) return null;
    const apiKey =
      typeof values["llm.api_key"] === "string" ? values["llm.api_key"] : "";
    const baseUrl =
      typeof values["llm.base_url"] === "string" ? values["llm.base_url"] : "";

    const name = deriveProfileNameFromModel(model);
    const llmConfig: { model: string; api_key?: string; base_url?: string } = {
      model,
    };
    if (apiKey) llmConfig.api_key = apiKey;
    if (baseUrl) llmConfig.base_url = baseUrl;

    try {
      await saveProfile.mutateAsync({
        name,
        request: { llm: llmConfig, include_secrets: true },
      });
      await activateProfile.mutateAsync(name);
      return name;
    } catch (error) {
      // Best-effort: the agent_settings save already succeeded, so the
      // user is not blocked from completing onboarding.
      console.error("Failed to persist onboarding LLM as profile:", error);
      return null;
    }
  }, [isLocalBackend, saveControl, saveProfile, activateProfile]);

  const handleSaveSuccess = React.useCallback(async () => {
    setIsFinalizing(true);
    try {
      const llmProfileName = await persistAsProfile();
      // Point the active AGENT profile at the LLM the user just configured so
      // the next conversation actually uses it (and the "LLM not set up"
      // banner clears). Without this the active agent profile keeps its
      // seeded llm_profile_ref, which has no key.
      //
      // Cloud intentionally skips this: `persistAsProfile` only creates/activates
      // a *local* LLM profile (it early-returns null off local backends), and on
      // cloud the agent-profile ↔ LLM wiring is resolved server-side from the
      // settings this step's form save already persisted — there is no
      // client-writable cloud agent-profile ref to repoint here. The ACP step
      // still calls applyAgentProfile because ACP agents carry no LLM ref and
      // persist their kind/model locally regardless of backend.
      if (llmProfileName) {
        await applyAgentProfile({
          agent_kind: "openhands",
          llm_profile_ref: llmProfileName,
        });
      }
    } finally {
      setIsFinalizing(false);
      onNext();
    }
  }, [persistAsProfile, applyAgentProfile, onNext]);

  const handleNext = () => {
    if (saveControl?.isDirty) {
      saveControl.save();
      // `onSaveSuccess` (wired to `handleSaveSuccess` below) will advance
      // once the mutation resolves successfully.
      return;
    }
    onNext();
  };

  return (
    <div
      data-testid="onboarding-step-setup-llm"
      className="flex flex-col gap-6 max-h-[calc(90vh-7rem)]"
    >
      <header className="flex flex-col gap-2">
        <h2 className="text-2xl font-medium text-white">
          {t(I18nKey.ONBOARDING$LLM_TITLE)}
        </h2>
        <p className="text-sm text-[var(--oh-muted)]">
          {t(I18nKey.ONBOARDING$LLM_SUBTITLE)}
        </p>
      </header>

      <div
        data-testid="onboarding-llm-settings"
        className="flex min-h-0 flex-1 flex-col overflow-y-auto custom-scrollbar-always"
      >
        <LlmSettingsScreen
          embedded
          hideSaveButton
          suppressSuccessToast
          initialValueOverrides={{
            "llm.model": ONBOARDING_DEFAULT_LLM_MODEL,
          }}
          onSaveSuccess={handleSaveSuccess}
          onSaveControlChange={setSaveControl}
        />
      </div>

      <div className="sticky bottom-0 flex items-center justify-between gap-2 bg-base-secondary pt-4 pb-7">
        <BrandButton
          testId="onboarding-llm-back"
          type="button"
          variant="secondary"
          onClick={onBack}
        >
          {t(I18nKey.ONBOARDING$BACK)}
        </BrandButton>
        <BrandButton
          testId="onboarding-llm-next"
          type="button"
          variant="primary"
          isDisabled={(saveControl?.isSaving ?? false) || isFinalizing}
          onClick={handleNext}
        >
          {t(I18nKey.ONBOARDING$NEXT)}
        </BrandButton>
      </div>
    </div>
  );
}
