import { useEffect, useRef } from "react";
import AgentProfilesService, {
  type AgentProfileListResponse,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import type { ProfileListResponse } from "#/api/profiles-service/profiles-service.api";
import { getAcpPreferredDefaultModel } from "#/constants/acp-providers";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useActivateAgentProfile } from "#/hooks/mutation/use-activate-agent-profile";
import { useActivateLlmProfile } from "#/hooks/mutation/use-activate-llm-profile";
import { useAgentProfiles } from "#/hooks/query/use-agent-profiles";
import { useHostDetectedConnections } from "#/hooks/query/use-host-detected-connections";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useActiveAgentProfile } from "#/hooks/use-active-agent-profile";
import { formatCommand } from "#/utils/acp-command";
import { isChatgptAutoProfileName } from "#/utils/subscription-model-catalog";
import {
  pickPreferredHostLaunch,
  type PreferredHostLaunch,
} from "#/utils/host-detected-provider-connections";
import { isUsableLlmProfile } from "#/utils/usable-llm-profile";

export const CLAUDE_CODE_AUTO_AGENT_PROFILE_NAME = "claude-code";

export async function applyPreferredHostLaunch(options: {
  launch: PreferredHostLaunch;
  llmProfiles: ProfileListResponse;
  agentProfiles: AgentProfileListResponse;
  activateLlm: (name: string) => void;
  activateAgent: (profileId: string) => void;
}): Promise<void> {
  const { launch, llmProfiles, agentProfiles, activateLlm, activateAgent } =
    options;

  if (launch.kind === "chatgpt") {
    const chatgpt = llmProfiles.profiles.find((profile) =>
      isChatgptAutoProfileName(profile.name),
    );
    if (chatgpt) activateLlm(chatgpt.name);
    return;
  }

  const named = agentProfiles.profiles.find(
    (profile) =>
      profile.agent_kind === "acp" && profile.name === launch.profileName,
  );
  let profileId = named?.id ?? null;
  if (!profileId && !launch.command) {
    for (const profile of agentProfiles.profiles) {
      if (profile.agent_kind !== "acp") continue;
      const detail = await AgentProfilesService.getProfile(profile.name);
      if (
        detail.profile.agent_kind === "acp" &&
        detail.profile.acp_server === launch.acpServer
      ) {
        profileId = profile.id;
        break;
      }
    }
  }
  if (!profileId) {
    await AgentProfilesService.saveProfile(launch.profileName, {
      agent_kind: "acp",
      acp_server: launch.acpServer,
      acp_command: launch.command ? formatCommand([...launch.command]) : null,
      acp_model: getAcpPreferredDefaultModel(launch.acpServer) ?? undefined,
    });
    const created = await AgentProfilesService.getProfile(launch.profileName);
    profileId = created.profile.id;
  }
  if (profileId) activateAgent(profileId);
}

/**
 * When OpenHands has no usable LLM but a host subscription can run chats,
 * activate that launch path (Claude Code, ChatGPT, Cursor CLI, or OpenCode).
 */
export function useEnsurePreferredSubscriptionLaunch(): void {
  const { backend } = useActiveBackend();
  const isLocal = backend.kind === "local";
  const { connections, isChecking } = useHostDetectedConnections();
  const { data: llmProfiles } = useLlmProfiles();
  const { data: agentProfiles, isLoading: agentProfilesLoading } =
    useAgentProfiles();
  const { activeProfile } = useActiveAgentProfile();
  const { mutate: activateAgent, isPending: activateAgentPending } =
    useActivateAgentProfile();
  const { mutate: activateLlm, isPending: activateLlmPending } =
    useActivateLlmProfile();
  const attemptedRef = useRef<string | null>(null);

  useEffect(() => {
    attemptedRef.current = null;
  }, [backend.id]);

  useEffect(() => {
    if (!isLocal || isChecking || agentProfilesLoading) return;
    if (!llmProfiles || !agentProfiles) return;
    if (activateAgentPending || activateLlmPending) return;
    if (activeProfile?.agent_kind === "acp") return;

    const activeLlm = llmProfiles.profiles.find(
      (profile) => profile.name === llmProfiles.active_profile,
    );
    if (isUsableLlmProfile(activeLlm)) return;

    const launch = pickPreferredHostLaunch(connections);
    if (!launch) return;

    const attemptKey =
      launch.kind === "chatgpt" ? "chatgpt" : launch.profileName;
    if (attemptedRef.current === attemptKey) return;
    attemptedRef.current = attemptKey;

    void applyPreferredHostLaunch({
      launch,
      llmProfiles,
      agentProfiles,
      activateLlm,
      activateAgent,
    }).catch(() => {
      attemptedRef.current = null;
    });
  }, [
    activateAgent,
    activateAgentPending,
    activateLlm,
    activateLlmPending,
    activeProfile?.agent_kind,
    agentProfiles,
    agentProfilesLoading,
    connections,
    isChecking,
    isLocal,
    llmProfiles,
  ]);
}
