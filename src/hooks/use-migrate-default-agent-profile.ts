import {
  useQuery,
  useQueryClient,
  type QueryClient,
} from "@tanstack/react-query";
import AgentProfilesService, {
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import type { StoredOpenHandsProfile } from "#/api/agent-profiles-service/profile-behavior";
import SettingsService from "#/api/settings-service/settings-service.api";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { mergeAgentProfileSaveInput } from "#/components/features/settings/agent-profiles/merge-agent-profile-save-input";
import {
  AGENT_PROFILES_QUERY_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";
import type { Settings } from "#/types/settings";
import {
  migrateSkillEnablement,
  resolveEnabledCatalogSkills,
  toSkillEnablement,
} from "#/utils/skill-enablement";

const PROFILE_VERIFICATION_FIELDS = [
  "critic_enabled",
  "critic_mode",
  "enable_iterative_refinement",
  "critic_threshold",
  "max_refinement_iterations",
  "critic_server_url",
  "critic_model_name",
] as const;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === "object" && !Array.isArray(value);

/**
 * The `default` profile fields that reproduce the global settings it used to
 * launch from. Skills come from the Skills page lists; the critic API key
 * stays global.
 */
export function buildDefaultProfileMigration(
  settings: Settings,
): Partial<StoredOpenHandsProfile> {
  const enablement = toSkillEnablement(settings);
  const skills = migrateSkillEnablement(enablement) ?? {
    enabled_skills: resolveEnabledCatalogSkills(enablement),
    disabled_skills: enablement.disabledSkills ?? [],
  };
  const migrated: Partial<StoredOpenHandsProfile> = {
    public_skills: skills.enabled_skills,
    disabled_skills: skills.disabled_skills,
  };

  // Global settings left on an ACP agent carry no OpenHands behavior.
  const agent: Record<string, unknown> = settings.agent_settings ?? {};
  if (agent.agent_kind === "acp") return migrated;

  const context = isRecord(agent.agent_context) ? agent.agent_context : {};
  const globalVerification = isRecord(agent.verification)
    ? agent.verification
    : null;
  // The secret-free subset; the critic API key stays global.
  const verification = globalVerification
    ? (Object.fromEntries(
        PROFILE_VERIFICATION_FIELDS.filter(
          (key) => key in globalVerification,
        ).map((key) => [key, globalVerification[key]]),
      ) as unknown as StoredOpenHandsProfile["verification"])
    : undefined;

  return {
    ...migrated,
    ...(isRecord(agent.condenser) ? { condenser: agent.condenser } : {}),
    ...(verification ? { verification } : {}),
    ...("tools" in agent
      ? { tools: (agent.tools ?? null) as StoredOpenHandsProfile["tools"] }
      : {}),
    ...(typeof agent.persona === "string" || agent.persona === null
      ? { persona: agent.persona }
      : {}),
    ...(typeof agent.agent === "string" ? { agent: agent.agent } : {}),
    ...(typeof agent.tool_concurrency_limit === "number"
      ? { tool_concurrency_limit: agent.tool_concurrency_limit }
      : {}),
    system_message_suffix:
      typeof context.system_message_suffix === "string"
        ? context.system_message_suffix
        : null,
    enable_classify_and_switch_llm_tool:
      agent.enable_classify_and_switch_llm_tool === true,
    meta_profile_ref:
      typeof agent.active_meta_profile === "string"
        ? agent.active_meta_profile
        : null,
  };
}

/** Copy global agent behavior into the OpenHands `default` profile, once. */
async function migrateDefaultAgentProfile(): Promise<true> {
  // Read the marker fresh: another tab may have migrated and then edited
  // `default` since this one cached its settings.
  SettingsService.invalidateCache();
  const settings = await SettingsService.getSettings();
  if (settings.default_profile_migrated) return true;

  const { profiles } = await AgentProfilesService.listProfiles();
  const hasOpenHandsDefault = profiles.some(
    (profile) =>
      profile.name === WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME &&
      profile.agent_kind === "openhands",
  );
  if (hasOpenHandsDefault) {
    const { profile } = await AgentProfilesService.getProfile(
      WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME,
    );
    if (profile.agent_kind === "openhands") {
      await AgentProfilesService.saveProfile(
        WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME,
        mergeAgentProfileSaveInput(profile, {
          agent_kind: "openhands",
          ...buildDefaultProfileMigration(settings),
        }),
      );
    }
  }
  await SettingsService.saveSettings({ default_profile_migrated: true });
  return true;
}

function migrationQuery(queryClient: QueryClient, backendId: string) {
  return {
    queryKey: ["default-profile-migration", backendId] as const,
    queryFn: async () => {
      await migrateDefaultAgentProfile();
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: AGENT_PROFILES_QUERY_KEYS.all,
        }),
        queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEYS.all }),
      ]);
      return true as const;
    },
    staleTime: Infinity,
    retry: false,
    meta: { disableToast: true },
  };
}

/**
 * Resolve once the local backend's `default` profile carries its migrated
 * settings. A failure is left for the next caller to retry.
 */
export function ensureDefaultAgentProfileMigrated(
  queryClient: QueryClient,
  backendId: string,
): Promise<true> {
  return queryClient.ensureQueryData(migrationQuery(queryClient, backendId));
}

/** Run the migration as soon as a local backend is active. */
export function useMigrateDefaultAgentProfile(): void {
  const queryClient = useQueryClient();
  const { backend } = useActiveBackend();
  useQuery({
    ...migrationQuery(queryClient, backend.id),
    enabled: backend.kind === "local" && !isNoBackend(backend),
  });
}
