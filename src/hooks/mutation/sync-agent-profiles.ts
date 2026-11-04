import type { QueryClient } from "@tanstack/react-query";
import { HttpError } from "@openhands/typescript-client";
import AgentProfilesService, {
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME,
  type AgentProfile,
  type AgentProfileSaveInput,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import {
  getActiveBackend,
  isNoBackend,
} from "#/api/backend-registry/active-store";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import { mergeAgentProfileSaveInput } from "#/components/features/settings/agent-profiles/merge-agent-profile-save-input";
import {
  AGENT_PROFILES_QUERY_KEYS,
  LLM_PROFILES_QUERY_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";
import { getSettingsQueryFn } from "#/hooks/query/use-settings";
import { settleRunningDefaultAgentProfileMigration } from "#/hooks/use-migrate-default-agent-profile";
import i18n from "#/i18n";
import { I18nKey } from "#/i18n/declaration";
import type { Settings } from "#/types/settings";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { getMcpServerEnabled, parseMcpConfig } from "#/utils/mcp-config";
import {
  profileSkillEnablement,
  sameSkillSelection,
  toProfileSkillFields,
  type SkillEnablement,
} from "#/utils/skill-enablement";

/** The fields to change on a profile, or `null` when it already matches. */
export type ProfileChange = (
  profile: AgentProfile,
) => Partial<AgentProfileSaveInput> | null;

let queue: Promise<void> = Promise.resolve();

async function readProfile(name: string): Promise<AgentProfile | null> {
  return AgentProfilesService.getProfile(name)
    .then((detail) => detail.profile)
    .catch((error: unknown) => {
      if (error instanceof HttpError && error.status === 404) return null;
      throw error;
    });
}

async function saveChange(
  name: string,
  stored: AgentProfile | null,
  change: ProfileChange,
): Promise<boolean> {
  const patch = stored ? change(stored) : null;
  if (!stored || !patch) return false;
  await AgentProfilesService.saveProfile(
    name,
    mergeAgentProfileSaveInput(stored, {
      agent_kind: stored.agent_kind,
      ...patch,
    } as AgentProfileSaveInput),
  );
  return true;
}

interface WriteOptions {
  /** Replaces the default failure toast, which says the global edit saved. */
  onError?: (error: unknown) => void;
  /** Reject instead, for a caller that must not go on after a failed write. */
  throwOnError?: boolean;
}

/** Run profile writes one at a time, so each reads the previous one's result. */
function enqueue(
  queryClient: QueryClient,
  write: () => Promise<boolean>,
  { onError, throwOnError = false }: WriteOptions = {},
): Promise<void> {
  const run = queue.then(async () => {
    const { backend } = getActiveBackend();
    if (isNoBackend(backend)) return;
    if (backend.kind === "local") {
      await settleRunningDefaultAgentProfileMigration(
        queryClient,
        backend.id,
      ).catch(() => undefined);
    }
    if (await write()) {
      await queryClient.invalidateQueries({
        queryKey: AGENT_PROFILES_QUERY_KEYS.all,
      });
    }
  });
  queue = run.catch(() => undefined);
  return run.catch((error: unknown) => {
    console.error("Failed to update agent profiles:", error);
    if (throwOnError) throw error;
    if (onError) onError(error);
    else
      displayErrorToast(i18n.t(I18nKey.SETTINGS$DEFAULT_PROFILE_NOT_UPDATED));
  });
}

const onLocalBackend =
  (write: () => Promise<boolean>) => async (): Promise<boolean> =>
    getActiveBackend().backend.kind === "local" && write();

/** Change one profile. */
export function updateAgentProfile(
  queryClient: QueryClient,
  name: string,
  change: ProfileChange,
  options?: WriteOptions,
): Promise<void> {
  return enqueue(
    queryClient,
    async () => saveChange(name, await readProfile(name), change),
    options,
  );
}

/** Change the local `default` profile. */
export function updateDefaultAgentProfile(
  queryClient: QueryClient,
  change: ProfileChange,
  options?: WriteOptions,
): Promise<void> {
  return enqueue(
    queryClient,
    onLocalBackend(async () =>
      saveChange(
        WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME,
        await readProfile(WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME),
        change,
      ),
    ),
    options,
  );
}

/** Apply a global page edit to every local profile. */
export function updateEveryAgentProfile(
  queryClient: QueryClient,
  change: ProfileChange,
  options?: WriteOptions,
): Promise<void> {
  return enqueue(
    queryClient,
    onLocalBackend(async () => {
      const { profiles } = await AgentProfilesService.listProfiles();
      const saved = await Promise.all(
        profiles.map(async ({ name }) =>
          saveChange(name, await readProfile(name), change),
        ),
      );
      return saved.some(Boolean);
    }),
    options,
  );
}

export const routerDetached =
  (name: string): ProfileChange =>
  (profile) =>
    profile.agent_kind === "openhands" &&
    (profile as { meta_profile_ref?: string | null }).meta_profile_ref === name
      ? ({
          meta_profile_ref: null,
          enable_classify_and_switch_llm_tool: false,
        } as Partial<AgentProfileSaveInput>)
      : null;

type McpRefs = { mcp_server_refs?: string[] | null };

export const mcpServerRenamed =
  (from: string, to: string): ProfileChange =>
  (profile) => {
    const refs = (profile as McpRefs).mcp_server_refs;
    return refs?.includes(from)
      ? { mcp_server_refs: refs.map((ref) => (ref === from ? to : ref)) }
      : null;
  };

export const mcpServerRemoved =
  (name: string): ProfileChange =>
  (profile) => {
    const refs = (profile as McpRefs).mcp_server_refs;
    return refs?.includes(name)
      ? { mcp_server_refs: refs.filter((ref) => ref !== name) }
      : null;
  };

export const withSkillSelection =
  (selection: SkillEnablement): ProfileChange =>
  (profile) => {
    if (profile.agent_kind !== "openhands") return null;
    return sameSkillSelection(profileSkillEnablement(profile), selection)
      ? null
      : toProfileSkillFields(selection);
  };

export const llmProfileMovedOff =
  (from: string, to: string): ProfileChange =>
  (profile) =>
    profile.agent_kind === "openhands" && profile.llm_profile_ref === from
      ? { llm_profile_ref: to }
      : null;

export const llmProfileMirrored =
  (name: string): ProfileChange =>
  (profile) =>
    profile.agent_kind === "openhands" && profile.llm_profile_ref !== name
      ? { llm_profile_ref: name }
      : null;

function mcpConfigOf(settings: Settings | undefined) {
  return (
    settings?.mcp_config ?? parseMcpConfig(settings?.agent_settings?.mcp_config)
  );
}

/** `default`'s MCP servers as the editor shows them: the ones switched on in MCP Servers. */
export function enabledMcpServerRefs(
  settings: Settings | undefined,
): string[] | null {
  const config = mcpConfigOf(settings);
  const enabled = Object.keys(config).filter(
    (key) => getMcpServerEnabled(config[key]) !== false,
  );
  return enabled.length === Object.keys(config).length ? null : enabled;
}

/** What an OpenHands `default` stores: every switched-on MCP server and every secret. */
export function sharedDefaultProfileFields(
  input: AgentProfileSaveInput,
): AgentProfileSaveInput {
  return {
    ...input,
    mcp_server_refs: null,
    ...("secret_refs" in input ? { secret_refs: null } : {}),
  } as AgentProfileSaveInput;
}

/** Make the active LLM profile and the MCP page's switches match an edited OpenHands `default`. */
export async function applyDefaultAgentProfileToGlobal(
  queryClient: QueryClient,
  input: AgentProfileSaveInput,
): Promise<void> {
  const settings = await getSettingsQueryFn();
  const writes: Promise<unknown>[] = [];

  if (input.agent_kind === "openhands" && input.llm_profile_ref) {
    const { active_profile: active } = await ProfilesService.listProfiles();
    if (active !== input.llm_profile_ref) {
      writes.push(ProfilesService.activateProfile(input.llm_profile_ref));
    }
  }

  const refs = (input as McpRefs).mcp_server_refs ?? null;
  const config = mcpConfigOf(settings);
  Object.keys(config).forEach((key) => {
    const enabled = refs === null || refs.includes(key);
    if ((getMcpServerEnabled(config[key]) !== false) !== enabled) {
      writes.push(SettingsService.patchMcpServer(key, { enabled }));
    }
  });

  await Promise.all(writes);
  if (writes.length > 0) {
    SettingsService.invalidateCache();
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEYS.all }),
      queryClient.invalidateQueries({ queryKey: LLM_PROFILES_QUERY_KEYS.all }),
    ]);
  }
}
