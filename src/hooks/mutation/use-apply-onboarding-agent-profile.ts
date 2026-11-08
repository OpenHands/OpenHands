import { useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { HttpError } from "@openhands/typescript-client";
import AgentProfilesService, {
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME,
  type AgentProfile,
  type AgentProfileSaveInput,
} from "#/api/agent-profiles-service/agent-profiles-service.api";
import { mergeAgentProfileSaveInput } from "#/components/features/settings/agent-profiles/merge-agent-profile-save-input";
import { useSaveAgentProfile } from "#/hooks/mutation/use-save-agent-profile";
import { useActivateAgentProfile } from "#/hooks/mutation/use-activate-agent-profile";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { ensureDefaultAgentProfileMigrated } from "#/hooks/use-migrate-default-agent-profile";

type AcpSaveInput = Extract<AgentProfileSaveInput, { agent_kind: "acp" }>;

export type OnboardingAgentProfile =
  | Extract<AgentProfileSaveInput, { agent_kind: "openhands" }>
  | (AcpSaveInput & { acp_server: NonNullable<AcpSaveInput["acp_server"]> });

const MAX_ACP_PROFILE_NAMES = 10;

async function readProfile(name: string): Promise<AgentProfile | null> {
  return AgentProfilesService.getProfile(name)
    .then((detail) => detail.profile)
    .catch((error: unknown) => {
      if (error instanceof HttpError && error.status === 404) return null;
      throw error;
    });
}

/** Find the provider's profile, or the first free name for it: `codex`, `codex-2`, … */
async function findAcpProfile(
  acpServer: string,
  attempt = 1,
): Promise<{ name: string; stored: AgentProfile | null }> {
  const name = attempt === 1 ? acpServer : `${acpServer}-${attempt}`;
  const stored = await readProfile(name);
  if (
    stored === null ||
    (stored.agent_kind === "acp" && stored.acp_server === acpServer)
  ) {
    return { name, stored };
  }
  if (attempt >= MAX_ACP_PROFILE_NAMES) {
    throw new Error(`No free agent profile name for ${acpServer}`);
  }
  return findAcpProfile(acpServer, attempt + 1);
}

/**
 * Configure and activate the agent profile for the user's onboarding choice:
 * OpenHands lands on `default`, an ACP provider on its own profile.
 *
 * Best-effort: a failure is swallowed so onboarding is never blocked; the
 * error shows when the user starts a conversation.
 */
export function useApplyOnboardingAgentProfile() {
  const queryClient = useQueryClient();
  const { backend } = useActiveBackend();
  const saveProfile = useSaveAgentProfile();
  const activateProfile = useActivateAgentProfile();

  return useCallback(
    async (profile: OnboardingAgentProfile) => {
      try {
        if (backend.kind === "local" && !isNoBackend(backend)) {
          // A failed migration retries later and merges over this save.
          await ensureDefaultAgentProfileMigrated(
            queryClient,
            backend.id,
          ).catch(() => undefined);
        }
        let name: string = WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME;
        let id: string | null | undefined;
        if (profile.agent_kind === "acp") {
          // A provider's existing profile keeps the user's model and command.
          const found = await findAcpProfile(profile.acp_server);
          name = found.name;
          id = found.stored?.id;
          if (!found.stored) await saveProfile.mutateAsync({ name, profile });
        } else {
          // Merge over the stored `default`: it carries the user's migrated
          // skills, condenser and verification, which re-onboarding must keep.
          const stored = await readProfile(name);
          await saveProfile.mutateAsync({
            name,
            profile: mergeAgentProfileSaveInput(stored, profile),
          });
        }
        // activate needs the stable id; the save response only echoes the name.
        id ??= (await AgentProfilesService.getProfile(name)).profile.id;
        if (id) await activateProfile.mutateAsync(id);
      } catch (error) {
        console.error("Failed to configure onboarding agent profile:", error);
      }
    },
    [queryClient, backend, saveProfile, activateProfile],
  );
}
