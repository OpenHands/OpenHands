import { useQuery } from "@tanstack/react-query";
import { useSettings } from "#/hooks/query/use-settings";
import { useConfig } from "#/hooks/query/use-config";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useActiveAgentProfile } from "#/hooks/use-active-agent-profile";
import { isSettingsPageHidden } from "#/utils/settings-utils";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import {
  CONFIG_CACHE_OPTIONS,
  LLM_PROFILES_QUERY_KEYS,
} from "#/hooks/query/query-keys";
import { isSubscriptionLlmConfig } from "#/constants/llm-subscription";

interface LlmConfiguredResult {
  /**
   * True when the active backend's agent has a usable LLM:
   * - ACP agents own their LLM via a subprocess, so they never need a key.
   * - OpenHands agents are ready only once an LLM API key has been saved.
   * - When the LLM settings page is hidden by a feature flag there is no
   *   place to finish setup, so we treat the LLM as configured to avoid
   *   surfacing an actionless warning.
   */
  isConfigured: boolean;
  /**
   * True while the configured/unconfigured state is indeterminate — either
   * settings/config are still resolving, or a fetch failed and left us with no
   * data to decide from. Consumers should render nothing in this state so a
   * warning doesn't flash before data loads or on a transient network error.
   */
  isLoading: boolean;
}

/**
 * Reports whether the active backend's agent has an LLM ready to run
 * conversations. Surfaces the gap left by the onboarding "Skip for now" path,
 * which persists no settings — leaving an OpenHands agent without an API key.
 */
export function useLlmConfigured(): LlmConfiguredResult {
  const {
    data: settings,
    isLoading: settingsLoading,
    isError: settingsError,
  } = useSettings();
  const {
    data: config,
    isLoading: configLoading,
    isError: configError,
  } = useConfig();
  const {
    data: profilesData,
    isLoading: profilesLoading,
    isError: profilesError,
  } = useLlmProfiles();
  const { backend, orgId } = useActiveBackend();
  const isLocal = backend.kind === "local";

  // The active AgentProfile is the current agent — an ACP profile owns its LLM
  // via the subprocess and never needs an API key.
  const {
    activeProfile: activeAgentProfile,
    agentKind: activeAgentKind,
    isLoading: activeAgentProfileLoading,
    isError: activeAgentProfileError,
  } = useActiveAgentProfile();
  const isAcpAgent = activeAgentKind === "acp";
  const hasApiKey = settings?.llm_api_key_set === true;
  // Check the LLM profile the next home launch runs. Locally that is the
  // active LLM profile (the home pill), which overrides the agent profile's
  // own `llm_profile_ref`; Cloud launches run the ref. Either falls back to
  // the other when it names no saved profile.
  const findLlmProfile = (name: string | null | undefined) =>
    name
      ? profilesData?.profiles.find((profile) => profile.name === name)
      : undefined;
  const referencedProfile = findLlmProfile(
    activeAgentProfile?.agent_kind === "openhands"
      ? activeAgentProfile.llm_profile_ref
      : undefined,
  );
  const selectedProfile = findLlmProfile(profilesData?.active_profile);
  const activeProfile = isLocal
    ? (selectedProfile ?? referencedProfile)
    : (referencedProfile ?? selectedProfile);
  const hasActiveProfileApiKey = activeProfile?.api_key_set === true;
  const shouldLoadActiveProfileDetail =
    isLocal && !!activeProfile && !hasActiveProfileApiKey;
  const {
    data: activeProfileDetail,
    isLoading: activeProfileDetailLoading,
    isError: activeProfileDetailError,
  } = useQuery({
    queryKey: [
      ...LLM_PROFILES_QUERY_KEYS.all,
      backend.id,
      orgId,
      "detail",
      activeProfile?.name,
    ],
    queryFn: () => ProfilesService.getProfile(activeProfile!.name),
    ...CONFIG_CACHE_OPTIONS,
    enabled: shouldLoadActiveProfileDetail,
    meta: { disableToast: true },
  });
  const hasActiveProfileSubscription =
    shouldLoadActiveProfileDetail &&
    isSubscriptionLlmConfig(
      activeProfileDetail?.config as Record<string, unknown> | undefined,
    );
  const llmSettingsHidden = isSettingsPageHidden(
    "/settings/llm",
    config?.feature_flags,
  );

  // Profiles are the source of truth on local. Cloud still accepts the legacy
  // settings-key signal, but provider-backed org profiles can be usable even
  // when settings.llm_api_key_set is false.
  const hasUsableActiveProfile =
    hasActiveProfileApiKey || hasActiveProfileSubscription;
  const hasUsableLlm = isLocal
    ? hasUsableActiveProfile
    : hasApiKey || hasUsableActiveProfile;

  // Treat a fetch failure as indeterminate (same as loading) only when it
  // leaves us with no data to decide from — otherwise a transient network
  // error would surface the banner with the same urgency as a genuinely
  // missing API key. A settings 404 is deliberately not covered here:
  // `useSettings` maps it to DEFAULT_SETTINGS (no key, OpenHands agent) while
  // keeping `isError` set, and that is exactly the new-user / "Skip for now"
  // state the banner exists to catch — so we keep deciding from that data.
  const settingsIndeterminate = settingsLoading || (settingsError && !settings);
  const configIndeterminate = configLoading || (configError && !config);
  // The active agent profile decides `isAcpAgent` and which LLM profile the
  // next conversation runs; until it is known, decide nothing — otherwise an
  // ACP agent (which needs no key) reads as an unconfigured OpenHands agent and
  // shows the "LLM not set up" banner.
  const agentProfileIndeterminate =
    activeAgentProfileLoading || activeAgentProfileError;
  const profilesIndeterminate =
    profilesLoading || (profilesError && !profilesData);
  const activeProfileDetailIndeterminate =
    shouldLoadActiveProfileDetail &&
    (activeProfileDetailLoading ||
      (activeProfileDetailError && !activeProfileDetail));

  return {
    isConfigured: isAcpAgent || llmSettingsHidden || hasUsableLlm,
    isLoading:
      settingsIndeterminate ||
      configIndeterminate ||
      profilesIndeterminate ||
      agentProfileIndeterminate ||
      activeProfileDetailIndeterminate,
  };
}
