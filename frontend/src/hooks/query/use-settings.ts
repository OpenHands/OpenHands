import { useQuery } from "@tanstack/react-query";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { useIsOnIntermediatePage } from "#/hooks/use-is-on-intermediate-page";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { Settings, SettingsScope, SettingsValue } from "#/types/settings";
import { organizationService } from "#/api/organization-service/organization-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import { useIsAuthed } from "./use-is-authed";
import { useConfig } from "./use-config";
import {
  pickFirstBoolean,
  pickFirstNumber,
  pickNullableString,
} from "#/utils/settings-value-pickers";
import { parseMcpConfig } from "#/utils/mcp-config";

/** Look up a value in a nested object by dotted key path. */
const lookupNested = (obj: Record<string, unknown>, key: string): unknown => {
  const parts = key.split(".");
  let current: unknown = obj;
  for (const part of parts) {
    if (current == null || typeof current !== "object") return undefined;
    current = (current as Record<string, unknown>)[part];
  }
  return current;
};

const resolveSdkString = (
  agentSettings: Record<string, unknown>,
  key: string,
  defaultValue: string,
  allowEmpty = false,
): string => {
  const value = lookupNested(agentSettings, key);
  if (typeof value === "string" && (value.length > 0 || allowEmpty)) {
    return value;
  }
  return defaultValue;
};

const normalizeSettingsResponse = (settings: Partial<Settings>): Settings => {
  const agentSettings = (settings.agent_settings ?? {}) as Record<
    string,
    unknown
  >;
  const conversationSettings = {
    ...(DEFAULT_SETTINGS.conversation_settings ?? {}),
    ...((settings.conversation_settings ?? {}) as Record<
      string,
      SettingsValue
    >),
  };

  return {
    ...DEFAULT_SETTINGS,
    ...settings,
    llm_model: resolveSdkString(
      agentSettings,
      "llm.model",
      DEFAULT_SETTINGS.llm_model,
    ),
    llm_base_url: resolveSdkString(
      agentSettings,
      "llm.base_url",
      DEFAULT_SETTINGS.llm_base_url,
      true,
    ),
    agent: resolveSdkString(agentSettings, "agent", DEFAULT_SETTINGS.agent),
    llm_api_key: settings.llm_api_key ?? null,
    llm_api_key_set: settings.llm_api_key_set ?? false,
    confirmation_mode:
      pickFirstBoolean(conversationSettings.confirmation_mode) ??
      DEFAULT_SETTINGS.confirmation_mode,
    security_analyzer:
      pickNullableString(conversationSettings.security_analyzer) ??
      DEFAULT_SETTINGS.security_analyzer,
    max_iterations:
      pickFirstNumber(conversationSettings.max_iterations) ??
      DEFAULT_SETTINGS.max_iterations,
    enable_default_condenser:
      pickFirstBoolean(lookupNested(agentSettings, "condenser.enabled")) ??
      DEFAULT_SETTINGS.enable_default_condenser,
    condenser_max_size:
      pickFirstNumber(lookupNested(agentSettings, "condenser.max_size")) ??
      DEFAULT_SETTINGS.condenser_max_size,
    mcp_config: parseMcpConfig(
      settings.mcp_config ??
        (agentSettings.mcp_config as typeof settings.mcp_config),
    ),
    search_api_key: settings.search_api_key || "",
    email: settings.email || "",
    git_user_name: settings.git_user_name || DEFAULT_SETTINGS.git_user_name,
    git_user_email: settings.git_user_email || DEFAULT_SETTINGS.git_user_email,
    is_new_user: false,
    disabled_skills:
      settings.disabled_skills ?? DEFAULT_SETTINGS.disabled_skills,
    v1_enabled: settings.v1_enabled ?? DEFAULT_SETTINGS.v1_enabled,
    agent_settings_schema: settings.agent_settings_schema ?? null,
    agent_settings: settings.agent_settings ?? DEFAULT_SETTINGS.agent_settings,
    conversation_settings_schema:
      settings.conversation_settings_schema ??
      DEFAULT_SETTINGS.conversation_settings_schema,
    conversation_settings: conversationSettings,
    sandbox_grouping_strategy:
      settings.sandbox_grouping_strategy ??
      DEFAULT_SETTINGS.sandbox_grouping_strategy,
  };
};

export const getSettingsQueryFn = async (
  scope: SettingsScope = "personal",
): Promise<Settings> => {
  const settings =
    scope === "org"
      ? await organizationService.getOrganizationAgentSettings()
      : await SettingsService.getSettings();

  return normalizeSettingsResponse(settings);
};

export const useSettings = (scope: SettingsScope = "personal") => {
  const isOnIntermediatePage = useIsOnIntermediatePage();
  const { data: userIsAuthenticated } = useIsAuthed();
  const { organizationId } = useSelectedOrganizationId();
  const { data: config } = useConfig();

  const isOss = config?.app_mode === "oss";

  const query = useQuery({
    queryKey: ["settings", scope, organizationId],
    queryFn: () => getSettingsQueryFn(scope),
    retry: (_, error) => error.status !== 404,
    refetchOnWindowFocus: false,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
    enabled:
      !isOnIntermediatePage &&
      !!userIsAuthenticated &&
      (isOss || !!organizationId),
    meta: {
      disableToast: true,
    },
  });

  // We want to return the defaults if the settings aren't found so the user can still see the
  // options to make their initial save. We don't set the defaults in `initialData` above because
  // that would prepopulate the data to the cache and mess with expectations. Read more:
  // https://tanstack.com/query/latest/docs/framework/react/guides/initial-query-data#using-initialdata-to-prepopulate-a-query
  if (query.error?.status === 404) {
    // Create a new object with only the properties we need, avoiding rest destructuring
    return {
      data: DEFAULT_SETTINGS,
      error: query.error,
      isError: query.isError,
      isLoading: query.isLoading,
      isFetching: query.isFetching,
      isFetched: query.isFetched,
      isSuccess: query.isSuccess,
      status: query.status,
      fetchStatus: query.fetchStatus,
      refetch: query.refetch,
    };
  }

  return query;
};
