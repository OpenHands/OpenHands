import React from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { LlmSettingsScreen } from "#/routes/llm-settings";
import type { SdkSectionSaveControl } from "#/components/features/settings/sdk-settings/sdk-section-page";
import {
  buildProfileLlmConfig,
  profileConfigToFormValues,
} from "#/components/features/settings/llm-profiles/llm-profile-form";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useSaveLlmProfile } from "#/hooks/mutation/use-save-llm-profile";
import { useActivateLlmProfile } from "#/hooks/mutation/use-activate-llm-profile";
import { useApplyOnboardingAgentProfile } from "#/hooks/mutation/use-apply-onboarding-agent-profile";
import {
  useDefaultModel,
  useDefaultModelReady,
} from "#/hooks/query/use-free-models";
import {
  LLM_PROFILES_QUERY_KEYS,
  useLlmProfiles,
} from "#/hooks/query/use-llm-profiles";
import { useSettings } from "#/hooks/query/use-settings";
import { useAgentSettingsSchema } from "#/hooks/query/use-agent-settings-schema";
import { LlmSettingsInputsSkeleton } from "#/components/features/settings/llm-settings/llm-settings-inputs-skeleton";
import { deriveProfileNameFromModel } from "#/utils/derive-profile-name";
import ProfilesService, {
  type SaveProfileRequest,
} from "#/api/profiles-service/profiles-service.api";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import type { SettingsFormValues } from "#/utils/sdk-settings-schema";

interface SetupLlmStepProps {
  onBack: () => void;
  onNext: () => void;
}

/**
 * Fallback when the backend has not exposed a DB-selected OpenHands default.
 * The onboarding override prefills the model, and Next saves it even when the
 * user leaves it as is.
 */
export const ONBOARDING_DEFAULT_LLM_MODEL = "openai/gpt-5.6-sol";

interface ProfileSeed {
  /** The active LLM profile's config (secrets encrypted), or `{}` without one. */
  baseConfig: Record<string, unknown>;
  /** The form values that show that profile. */
  initialValues: SettingsFormValues;
}

/**
 * The profile the local LLM step edits: the active LLM profile, or a fresh one
 * when none is active. Raw LLM settings are never the source. The seed is
 * taken once, when the profile list, the profile and the schema have loaded,
 * because the embedded form reads its initial values only when it mounts.
 */
function useActiveProfileSeed(enabled: boolean): ProfileSeed | null {
  const { backend, orgId } = useActiveBackend();
  const [seed, setSeed] = React.useState<ProfileSeed | null>(null);
  const profiles = useLlmProfiles({ enabled });
  const activeName = profiles.data?.active_profile;
  const activeProfileName =
    activeName &&
    profiles.data?.profiles.some((profile) => profile.name === activeName)
      ? activeName
      : null;
  const activeConfig = useQuery({
    queryKey: [
      ...LLM_PROFILES_QUERY_KEYS.all,
      "config",
      backend.id,
      orgId,
      activeProfileName,
    ],
    queryFn: async () => {
      const detail = await ProfilesService.getProfile(
        activeProfileName as string,
        "encrypted",
      );
      return (detail.config ?? {}) as Record<string, unknown>;
    },
    // Read once: the seed never changes, and saving or activating the new
    // profile would otherwise refetch the secrets for nothing.
    enabled: enabled && !!activeProfileName && !seed,
    // Keep the encrypted secrets out of the cache once the step unmounts.
    gcTime: 0,
    retry: 1,
    meta: { disableToast: true },
  });
  const { data: settings } = useSettings();
  const { data: schema, error: schemaError } = useAgentSettingsSchema(
    settings?.agent_settings_schema,
  );

  const isProfileListSettled = profiles.isSuccess || profiles.isError;
  const isActiveConfigSettled =
    !activeProfileName || activeConfig.isSuccess || activeConfig.isError;
  const isSchemaSettled = !!schema || !!schemaError;

  React.useEffect(() => {
    if (!enabled || seed) return;
    if (!isProfileListSettled || !isActiveConfigSettled || !isSchemaSettled) {
      return;
    }
    // An unreadable active profile also starts a fresh one: the form then shows
    // no key, so the user can see that one is needed.
    const baseConfig = activeConfig.data ?? {};
    setSeed({
      baseConfig,
      initialValues: profileConfigToFormValues(schema, baseConfig),
    });
  }, [
    enabled,
    seed,
    isProfileListSettled,
    isActiveConfigSettled,
    isSchemaSettled,
    activeConfig.data,
    schema,
  ]);

  return seed;
}

/**
 * Step 2: embed the LLM settings form. The screen runs in `embedded`
 * mode (so it doesn't render its own sticky Save bar) and with
 * `hideSaveButton` set, surfacing its save state via
 * `onSaveControlChange`. We then render a single Next button at the
 * modal footer level matching the other onboarding steps.
 *
 * On a local backend the step edits an LLM profile, like the profile editor
 * in Settings: the form shows the active profile (or a fresh one), and Next
 * saves the form as a profile named after the model, activates it and points
 * the `default` agent profile at it. Activation applies the profile to
 * `agent_settings.llm` on the server, so the step never reads or writes raw
 * LLM settings.
 *
 * On Cloud, Next saves the form to the settings and `onSaveSuccess` advances.
 * The agent-profile ↔ LLM wiring is resolved server-side from those settings,
 * and there is no client-writable cloud agent-profile ref to repoint here. If
 * the form is untouched, Next advances without a save call.
 *
 * Note: returning Cloud users who already have an LLM configured are
 * intercepted upstream by `OnboardingHost`, so they never reach this
 * step. Users who do reach it are first-time installs (Cloud or Local)
 * who want the OpenHands default pre-filled.
 */
export function SetupLlmStep({ onBack, onNext }: SetupLlmStepProps) {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const isLocalBackend = backend.kind === "local";
  const saveProfile = useSaveLlmProfile();
  const activateProfile = useActivateLlmProfile();
  const applyAgentProfile = useApplyOnboardingAgentProfile();
  const dbDefaultLlmModel = useDefaultModel();
  const isDefaultModelReady = useDefaultModelReady();
  const defaultLlmModel = dbDefaultLlmModel ?? ONBOARDING_DEFAULT_LLM_MODEL;
  const profileSeed = useActiveProfileSeed(isLocalBackend);
  const [saveControl, setSaveControl] =
    React.useState<SdkSectionSaveControl | null>(null);
  const [isFinalizing, setIsFinalizing] = React.useState(false);

  const initialValueOverrides = React.useMemo(
    () => ({
      ...(profileSeed?.initialValues ?? {}),
      "llm.model": defaultLlmModel,
    }),
    [profileSeed, defaultLlmModel],
  );

  const persistProfile = React.useCallback(async () => {
    if (!saveControl || !profileSeed) return;

    let dirtyLlm: Record<string, unknown>;
    try {
      dirtyLlm = {
        ...((saveControl.getDirtyPayload().llm ?? {}) as Record<
          string,
          unknown
        >),
      };
    } catch (error) {
      displayErrorToast(
        error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
      );
      return;
    }
    // The form shows the onboarding model even when the user keeps it, so the
    // profile gets that model whether or not the field was edited.
    if (!Object.prototype.hasOwnProperty.call(dirtyLlm, "model")) {
      dirtyLlm.model = String(saveControl.values["llm.model"] ?? "");
    }
    const { llmConfig } = buildProfileLlmConfig({
      baseConfig: profileSeed.baseConfig,
      dirtyLlm,
      values: saveControl.values,
      view: saveControl.view,
      supportsConnections: true,
      isCloud: false,
    });
    const model = typeof llmConfig.model === "string" ? llmConfig.model : "";
    if (!model) {
      displayErrorToast(t(I18nKey.SETTINGS$MODEL_REQUIRED));
      return;
    }
    const profileName = deriveProfileNameFromModel(model);

    setIsFinalizing(true);
    try {
      await saveProfile.mutateAsync({
        name: profileName,
        request: {
          llm: llmConfig as SaveProfileRequest["llm"],
          include_secrets: true,
        },
      });
      await activateProfile.mutateAsync(profileName);
      // Conversations launch from the active AGENT profile, so point it at the
      // LLM the user just configured (this also clears the "LLM not set up"
      // banner). Otherwise it keeps its seeded llm_profile_ref, which has no
      // key.
      await applyAgentProfile({
        agent_kind: "openhands",
        llm_profile_ref: profileName,
      });
      onNext();
    } catch {
      // Nothing advanced, so Next retries the whole save.
      displayErrorToast(t(I18nKey.ERROR$GENERIC));
    } finally {
      setIsFinalizing(false);
    }
  }, [
    saveControl,
    profileSeed,
    saveProfile,
    activateProfile,
    applyAgentProfile,
    onNext,
    t,
  ]);

  const handleNext = () => {
    if (isLocalBackend) {
      void persistProfile();
      return;
    }
    if (saveControl?.isDirty) {
      // `onSaveSuccess` advances once the settings save resolves.
      saveControl.save();
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
        <h2 className="text-2xl font-medium text-contrast">
          {t(I18nKey.ONBOARDING$LLM_TITLE)}
        </h2>
        <p className="text-sm text-muted">
          {t(I18nKey.ONBOARDING$LLM_SUBTITLE)}
        </p>
      </header>

      <div
        data-testid="onboarding-llm-settings"
        className="flex min-h-0 flex-1 flex-col overflow-y-auto custom-scrollbar-always"
      >
        {isDefaultModelReady && (!isLocalBackend || profileSeed) ? (
          <LlmSettingsScreen
            embedded
            hideSaveButton
            suppressSuccessToast
            initialValueOverrides={initialValueOverrides}
            // Local edits a profile, so like the profile editor only real
            // changes are dirty. Cloud saves the model override to settings,
            // so the override starts dirty there.
            markInitialOverridesDirty={!isLocalBackend}
            onSaveSuccess={isLocalBackend ? undefined : onNext}
            onSaveControlChange={setSaveControl}
          />
        ) : (
          <LlmSettingsInputsSkeleton />
        )}
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
          isDisabled={
            !isDefaultModelReady ||
            (isLocalBackend && !saveControl) ||
            (saveControl?.isSaving ?? false) ||
            isFinalizing
          }
          onClick={handleNext}
        >
          {t(I18nKey.ONBOARDING$NEXT)}
        </BrandButton>
      </div>
    </div>
  );
}
