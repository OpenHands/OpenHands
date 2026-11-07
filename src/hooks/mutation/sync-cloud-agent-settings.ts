import type { AxiosError } from "axios";
import AgentProfilesService, {
  type AgentProfile,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import SettingsService from "#/api/settings-service/settings-service.api";
import i18n from "#/i18n";
import { I18nKey } from "#/i18n/declaration";
import type { SettingsValue } from "#/types/settings";
import { parseCommand } from "#/utils/acp-command";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

function agentSettingsDiff(
  profile: AgentProfile,
): Record<string, SettingsValue> {
  if (profile.agent_kind === "openhands") return { agent_kind: "openhands" };
  // A kind flip resets the other acp_* fields, so send them all.
  return {
    agent_kind: "acp",
    acp_server: profile.acp_server,
    acp_command: profile.acp_command ? parseCommand(profile.acp_command) : [],
    acp_args: profile.acp_args ?? [],
    acp_model: profile.acp_model,
  };
}

/** Match Cloud `agent_settings` to the active agent profile (#18883). */
export async function syncCloudAgentSettings(
  savedName?: string,
): Promise<boolean> {
  if (getActiveBackend().backend.kind !== "cloud") return false;
  try {
    const { profiles, active_agent_profile_id: activeId } =
      await AgentProfilesService.listProfiles();
    const active = profiles.find((p) => p.id != null && p.id === activeId);
    if (!active || (savedName !== undefined && active.name !== savedName)) {
      return false;
    }
    const { profile } = await AgentProfilesService.getProfile(active.name);
    await SettingsService.saveSettings({
      agent_settings_diff: agentSettingsDiff(profile),
    });
    return true;
  } catch (error) {
    displayErrorToast(
      retrieveAxiosErrorMessage(error as AxiosError) ||
        i18n.t(I18nKey.ERROR$GENERIC),
    );
    return false;
  }
}
