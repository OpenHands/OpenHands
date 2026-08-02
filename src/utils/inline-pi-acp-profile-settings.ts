import type { ACPAgentProfile } from "@openhands/typescript-client";
import {
  adaptPiAcpCommandForDeployment,
  effectivePiAcpCommandTokens,
  PI_ACP_PROVIDER_KEY,
  resolveUiAcpProviderKey,
} from "#/constants/acp-providers";
import { getDeploymentMode } from "#/api/agent-server-adapter";
import { parseCommand } from "#/utils/acp-command";
import type { SettingsValue } from "#/types/settings";

/**
 * Pi agent profiles are stored as ``acp_server: "custom"`` with an explicit
 * ``pi-acp`` command. Launching via ``agent_profile_id`` makes the agent-server
 * spawn that command verbatim — bypassing the frontend ``resolveAcpCommand``
 * Docker rewrite — so profile-launched Pi conversations must be inlined as
 * ``agent_settings`` with a deployment-adapted argv.
 */
export function buildInlinePiAcpAgentSettingsFromProfile(
  profile: ACPAgentProfile,
  deploymentMode: string | null | undefined = getDeploymentMode(),
): Record<string, SettingsValue> | null {
  if (profile.agent_kind !== "acp") return null;

  const commandTokens = profile.acp_command
    ? parseCommand(profile.acp_command)
    : [];
  const uiKey = resolveUiAcpProviderKey(profile.acp_server, commandTokens);
  if (uiKey !== PI_ACP_PROVIDER_KEY) return null;

  const spawnTokens =
    commandTokens.length > 0
      ? commandTokens
      : effectivePiAcpCommandTokens(deploymentMode);
  const adaptedCommand = adaptPiAcpCommandForDeployment(
    spawnTokens,
    deploymentMode,
  );

  return {
    schema_version: 1,
    agent_kind: "acp",
    acp_server: profile.acp_server,
    acp_command: adaptedCommand as SettingsValue,
    // Pi owns model selection via ~/.pi; omit Canvas placeholders like "default".
    acp_model:
      profile.acp_model && profile.acp_model.trim() !== "default"
        ? profile.acp_model
        : null,
    acp_args: profile.acp_args ?? [],
  };
}
