import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { Settings, SettingsScope, SettingsValue } from "#/types/settings";
import SettingsService from "#/api/settings-service/settings-service.api";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";
import {
  pickFirstBoolean,
  pickFirstNumber,
  pickNullableString,
} from "#/utils/settings-value-pickers";
import { parseMcpConfig } from "#/utils/mcp-config";

export const getErrorStatus = (error: unknown): number | undefined => {
  if (typeof error === "object" && error !== null && "status" in error) {
    const { status } = error as { status?: unknown };
    if (typeof status === "number") {
      return status;
    }
  }

  if (axios.isAxiosError(error)) {
    return error.response?.status;
  }

  return undefined;
};

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

  const mcpConfig =
    agentSettings.mcp_config !== undefined
      ? parseMcpConfig(agentSettings.mcp_config)
      : (settings.mcp_config ?? DEFAULT_SETTINGS.mcp_config);

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
    mcp_config: mcpConfig,
    search_api_key: settings.search_api_key || "",
    email: settings.email || "",
    git_user_name: settings.git_user_name || DEFAULT_SETTINGS.git_user_name,
    git_user_email: settings.git_user_email || DEFAULT_SETTINGS.git_user_email,
    is_new_user: false,
    disabled_skills:
      settings.disabled_skills ?? DEFAULT_SETTINGS.disabled_skills,
    disabled_agents:
      settings.disabled_agents ?? DEFAULT_SETTINGS.disabled_agents,
    agent_settings_schema: settings.agent_settings_schema ?? null,
    agent_settings: settings.agent_settings ?? DEFAULT_SETTINGS.agent_settings,
    conversation_settings_schema:
      settings.conversation_settings_schema ??
      DEFAULT_SETTINGS.conversation_settings_schema,
    conversation_settings: conversationSettings,
  };
};

export const getSettingsQueryFn = async (
  scope: SettingsScope = "personal",
): Promise<Settings> => {
  if (scope !== "personal") {
    throw new Error(`Unsupported settings scope: ${scope}`);
  }

  const settings = await SettingsService.getSettings();
  return normalizeSettingsResponse(settings);
};

export const useSettings = (scope: SettingsScope = "personal") => {
  const active = useActiveBackend();
  const hasBackend = !isNoBackend(active.backend);
  const query = useQuery({
    // Include the active backend identity so switching backends or orgs
    // produces a fresh query — the `staleTime` cache for one backend
    // never serves another's data.
    queryKey: [
      ...SETTINGS_QUERY_KEYS.byScope(scope),
      active.backend.id,
      active.orgId,
    ],
    queryFn: () => getSettingsQueryFn(scope),
    retry: (_, error) => getErrorStatus(error) !== 404,
    enabled: hasBackend,
    refetchOnWindowFocus: false,
    staleTime: 1000 * 60 * 5,
    gcTime: 1000 * 60 * 15,
    meta: {
      disableToast: true,
    },
  });

  if (!hasBackend) {
    return {
      data: DEFAULT_SETTINGS,
      error: null,
      isError: false,
      isLoading: false,
      isFetching: false,
      isFetched: false,
      isSuccess: true,
      status: "success" as const,
      fetchStatus: "idle" as const,
      refetch: query.refetch,
    };
  }

  if (getErrorStatus(query.error) === 404) {
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
