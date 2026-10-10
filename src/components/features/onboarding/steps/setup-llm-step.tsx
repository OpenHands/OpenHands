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
import { isNoBackend } from "#/api/backend-registry/active-store";
import AgentProfilesService, {
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useSaveLlmProfile } from "#/hooks/mutation/use-save-llm-profile";
import { useActivateLlmProfile } from "#/hooks/mutation/use-activate-llm-profile";
import { useApplyOnboardingAgentProfile } from "#/hooks/mutation/use-apply-onboarding-agent-profile";
import {
  useDefaultModel,
  useDefaultModelReady,
} from "#/hooks/query/use-free-models";
import { LLM_PROFILES_QUERY_KEYS } from "#/hooks/query/use-llm-profiles";
import { useOpenAISubscriptionModels } from "#/hooks/query/use-llm-subscription-models";
import { useSettings } from "#/hooks/query/use-settings";
import { useAgentSettingsSchema } from "#/hooks/query/use-agent-settings-schema";
import { LlmSettingsInputsSkeleton } from "#/components/features/settings/llm-settings/llm-settings-inputs-skeleton";
import { deriveProfileNameFromModel } from "#/utils/derive-profile-name";
import ProfilesService, {
  type SaveProfileRequest,
} from "#/api/profiles-service/profiles-service.api";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import type { SettingsFormValues } from "#/utils/sdk-settings-schema";
import {
  LLM_AUTH_TYPE_KEY,
  LLM_AUTH_TYPE_SUBSCRIPTION,
  resolveLlmAuthType,
} from "#/constants/llm-subscription";

interface SetupLlmStepProps {
  onBack: () => void;
  onNext: () => void;
}

/**
 * Fallback when the backend has not exposed a DB-selected OpenHands default.
 * A fresh profile starts with this model, and Next saves it even when the user
 * leaves it as is. An existing profile keeps its own model.
 */
export const ONBOARDING_DEFAULT_LLM_MODEL = "openai/gpt-5.6-sol";

interface ProfileSeed {
  /** The existing LLM profile the form shows, or `null` for a fresh profile. */
  profileName: string | null;
  /** That profile's config (secrets encrypted), or `{}` for a fresh profile. */
  baseConfig: Record<string, unknown>;
  /** The form values that show that profile. */
  initialValues: SettingsFormValues;
}

/**
 * The LLM profile onboarding edits: the active LLM profile, else the one the
 * `default` agent profile points at, else none (a fresh profile). Listing the
 * agent profiles first also runs the Agent Server's one-time backfill, which
 * turns an older install's raw LLM settings into an LLM profile named
 * `default` and points the `default` agent profile at it, without activating
 * it. The active profile comes first because a home launch runs it.
 */
async function loadOnboardingProfile(): Promise<{
  profileName: string | null;
  config: Record<string, unknown>;
}> {
  const agentProfiles = await AgentProfilesService.listProfiles();
  const llmProfiles = await ProfilesService.listProfiles();
  const existing = new Set(llmProfiles.profiles.map((profile) => profile.name));
  const defaultAgentRef = agentProfiles.profiles.find(
    (profile) => profile.name === WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME,
  )?.llm_profile_ref;
  const profileName =
    [llmProfiles.active_profile, defaultAgentRef].find(
      (name): name is string => !!name && existing.has(name),
    ) ?? null;
  if (!profileName) return { profileName: null, config: {} };
  const detail = await ProfilesService.getProfile(profileName, "encrypted");
  return {
    profileName,
    config: (detail.config ?? {}) as Record<string, unknown>,
  };
}

/**
 * The profile the local LLM step edits (see `loadOnboardingProfile`). Raw LLM
 * settings are never the source. The seed is taken once per backend, because
 * the embedded form reads its initial values only when it mounts. Onboarding
 * mounts this step before a public-mode user adds the backend, so there is no
 * seed until a backend exists. A failed read is an error to retry, not a fresh
 * profile: a fresh form would drop an endpoint the Basic view hides.
 */
function useOnboardingProfileSeed(isLocalBackend: boolean): {
  seed: ProfileSeed | null;
  isError: boolean;
  retry: () => void;
} {
  const { backend, orgId } = useActiveBackend();
  const enabled = isLocalBackend && !isNoBackend(backend);
  const [seedState, setSeedState] = React.useState<{
    backendId: string;
    seed: ProfileSeed;
  } | null>(null);
  const seed = seedState?.backendId === backend.id ? seedState.seed : null;
  const profile = useQuery({
    queryKey: [
      ...LLM_PROFILES_QUERY_KEYS.all,
      "onboarding-seed",
      backend.id,
      orgId,
    ],
    queryFn: loadOnboardingProfile,
    // Read once: the seed never changes, and saving or activating the new
    // profile would otherwise refetch the secrets for nothing.
    enabled: enabled && !seed,
    // Keep the encrypted secrets out of the cache once the step unmounts.
    gcTime: 0,
    retry: 1,
    meta: { disableToast: true },
  });
  const { data: settings } = useSettings();
  const { data: schema, error: schemaError } = useAgentSettingsSchema(
    settings?.agent_settings_schema,
  );
  const isSchemaSettled = !!schema || !!schemaError;

  React.useEffect(() => {
    if (!enabled || seed || !profile.data || !isSchemaSettled) return;
    setSeedState({
      backendId: backend.id,
      seed: {
        profileName: profile.data.profileName,
        baseConfig: profile.data.config,
        initialValues: profileConfigToFormValues(schema, profile.data.config),
      },
    });
  }, [enabled, seed, backend.id, profile.data, isSchemaSettled, schema]);

  const { refetch } = profile;
  const retry = React.useCallback(() => {
    void refetch();
  }, [refetch]);

  return {
    seed,
    isError: enabled && !seed && profile.isError && !profile.isFetching,
    retry,
  };
}

/**
 * Step 2: embed the LLM settings form. The screen runs in `embedded`
 * mode (so it doesn't render its own sticky Save bar) and with
 * `hideSaveButton` set, surfacing its save state via
 * `onSaveControlChange`. We then render a single Next button at the
 * modal footer level matching the other onboarding steps.
 *
 * On a local backend the step edits an LLM profile, like the profile editor
 * in Settings: the form shows the profile `loadOnboardingProfile` picks, or a
 * fresh one with the default model. If the user changes nothing, Next keeps
 * that profile; otherwise it saves the form as a profile named after the
 * model. Either way it activates the profile and points the `default` agent
 * profile at it. Activation applies the profile to `agent_settings.llm` on the
 * server, so the step never reads or writes raw LLM settings.
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
  const {
    seed: profileSeed,
    isError: isProfileSeedError,
    retry: retryProfileSeed,
  } = useOnboardingProfileSeed(isLocalBackend);
  const [saveControl, setSaveControl] =
    React.useState<SdkSectionSaveControl | null>(null);
  const [isFinalizing, setIsFinalizing] = React.useState(false);
  const isSubscriptionAuth =
    resolveLlmAuthType(saveControl?.values[LLM_AUTH_TYPE_KEY]) ===
    LLM_AUTH_TYPE_SUBSCRIPTION;
  // Shares the form's query, so this reads the list its picker shows.
  const { data: subscriptionModels } = useOpenAISubscriptionModels({
    enabled: isLocalBackend && isSubscriptionAuth,
  });

  // An existing profile keeps its own model; only a fresh one gets the
  // onboarding default, so the form never pairs the default model with
  // another profile's key and endpoint.
  const initialValueOverrides = React.useMemo(() => {
    if (profileSeed?.profileName) return profileSeed.initialValues;
    return {
      ...(profileSeed?.initialValues ?? {}),
      "llm.model": defaultLlmModel,
    };
  }, [profileSeed, defaultLlmModel]);

  const activateForOnboarding = React.useCallback(
    async (profileName: string) => {
      await activateProfile.mutateAsync(profileName);
      // Conversations launch from the active AGENT profile, so point it at the
      // LLM the user just set up or kept (this also clears the "LLM not set
      // up" banner). Otherwise it keeps its seeded llm_profile_ref, which may
      // have no key.
      await applyAgentProfile({
        agent_kind: "openhands",
        llm_profile_ref: profileName,
      });
      onNext();
    },
    [activateProfile, applyAgentProfile, onNext],
  );

  const persistProfile = React.useCallback(async () => {
    if (!saveControl || !profileSeed) return;

    // Unchanged existing profile: keep it as it is rather than saving a copy
    // under the name derived from its model.
    if (profileSeed.profileName && !saveControl.isDirty) {
      setIsFinalizing(true);
      try {
        await activateForOnboarding(profileSeed.profileName);
      } catch {
        displayErrorToast(t(I18nKey.ERROR$GENERIC));
      } finally {
        setIsFinalizing(false);
      }
      return;
    }

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
    // The form shows its model even when the user keeps it (a fresh profile's
    // is the onboarding default), so the profile gets that model whether or
    // not the field was edited.
    if (!Object.prototype.hasOwnProperty.call(dirtyLlm, "model")) {
      dirtyLlm.model = String(saveControl.values["llm.model"] ?? "");
    }
    if (isSubscriptionAuth && !subscriptionModels?.length) {
      displayErrorToast("Subscription models are not loaded yet.");
      return;
    }
    const { llmConfig } = buildProfileLlmConfig({
      baseConfig: profileSeed.baseConfig,
      dirtyLlm,
      values: saveControl.values,
      view: saveControl.view,
      supportsConnections: true,
      isCloud: false,
      subscriptionModels,
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
      await activateForOnboarding(profileName);
    } catch {
      // Nothing advanced, so Next retries the whole save.
      displayErrorToast(t(I18nKey.ERROR$GENERIC));
    } finally {
      setIsFinalizing(false);
    }
  }, [
    saveControl,
    profileSeed,
    isSubscriptionAuth,
    subscriptionModels,
    saveProfile,
    activateForOnboarding,
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
          {t(
            profileSeed?.profileName
              ? I18nKey.ONBOARDING$LLM_SUBTITLE_CURRENT_PROFILE
              : I18nKey.ONBOARDING$LLM_SUBTITLE,
          )}
        </p>
      </header>

      <div
        data-testid="onboarding-llm-settings"
        className="flex min-h-0 flex-1 flex-col overflow-y-auto custom-scrollbar-always"
      >
        {isProfileSeedError ? (
          <div
            data-testid="onboarding-llm-load-error"
            className="flex flex-col items-start gap-3"
          >
            <p className="text-sm text-muted">
              {t(I18nKey.ERROR$FAILED_TO_LOAD_PROFILE_TRY_AGAIN)}
            </p>
            <BrandButton
              testId="onboarding-llm-load-retry"
              type="button"
              variant="secondary"
              onClick={retryProfileSeed}
            >
              {t(I18nKey.AUTOMATIONS$ERROR_RETRY)}
            </BrandButton>
          </div>
        ) : isDefaultModelReady && (!isLocalBackend || profileSeed) ? (
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
            (isLocalBackend && (!profileSeed || !saveControl)) ||
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
