import type { AgentProfileSummary } from "@openhands/typescript-client";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import type { Backend } from "#/api/backend-registry/types";
import {
  getCloudAgentProfile,
  listCloudAgentProfiles,
} from "#/api/cloud/agent-profiles-service.api";
import { fetchCloudProfilesForBackend } from "#/api/cloud/profiles-service.api";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import type { ChildConversationTarget } from "#/constants/child-conversation";
import type { AgentKind } from "#/types/settings";

/**
 * A saved agent profile the child launch will pin.
 *
 * `secretRefsRestricted` is true when the stored profile sets `secret_refs`
 * (including an empty list). The agent server then keeps only those secrets
 * because the launch sends `agent_profile_id` and does not fall back to
 * `agent_settings`.
 */
export interface ResolvedChildAgentProfile {
  id: string;
  name: string;
  agentKind: AgentKind;
  /** Model id the profile pins. Null when an ACP profile leaves the provider default. */
  model: string | null;
  llmProfileRef: string | null;
  secretRefsRestricted: boolean;
  /**
   * Why `model` is null for a selected profile. Set when a cloud agent profile
   * exists but its LLM profile is not on the list this session can see.
   */
  modelNote?: string;
}

export type ChildAgentProfileResolution =
  | { ok: true; profile: ResolvedChildAgentProfile }
  | { ok: false; error: string; guidance: string };

const rejected = (
  error: string,
  guidance: string,
): ChildAgentProfileResolution => ({
  ok: false,
  error,
  guidance,
});

const errorDetail = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

const savedProfileList = (profiles: readonly AgentProfileSummary[]) => {
  if (profiles.length === 0) return "there are no saved agent profiles";
  return profiles
    .map((profile) =>
      profile.id ? `"${profile.name}" (${profile.id})` : `"${profile.name}"`,
    )
    .join(", ");
};

const matchProfile = (
  profiles: readonly AgentProfileSummary[],
  selector: string,
) =>
  profiles.find((profile) => profile.id === selector) ??
  profiles.find((profile) => profile.name === selector) ??
  null;

async function listProfiles(
  target: ChildConversationTarget,
  cloudBackend: Backend | null,
) {
  if (target === "cloud") {
    if (!cloudBackend) {
      throw new Error("Cloud agent profiles require a cloud backend.");
    }
    return listCloudAgentProfiles(cloudBackend);
  }
  return AgentProfilesService.listProfiles();
}

async function readProfile(
  target: ChildConversationTarget,
  cloudBackend: Backend | null,
  name: string,
) {
  if (target === "cloud") {
    if (!cloudBackend) {
      throw new Error("Cloud agent profiles require a cloud backend.");
    }
    return getCloudAgentProfile(name, cloudBackend);
  }
  return AgentProfilesService.getProfile(name);
}

async function openHandsModel(
  target: ChildConversationTarget,
  cloudBackend: Backend | null,
  llmProfileRef: string,
): Promise<string | null | undefined> {
  const listed =
    target === "cloud"
      ? await fetchCloudProfilesForBackend(cloudBackend!)
      : await ProfilesService.listProfiles();
  const match = listed.profiles.find(
    (profile) => profile.name === llmProfileRef,
  );
  if (!match) return undefined;
  return match.model ?? null;
}

/**
 * Why a cloud child can still launch when this session cannot see the model.
 *
 * A local parent reaches a connected cloud backend that is not the active
 * org, so LLM confirmation uses the per-user settings route. An
 * organization-only LLM profile is absent from that route even when the
 * saved agent profile itself is valid. Cloud resolves `agent_profile_id`
 * server-side, so a missing visible model is reported, not treated as a
 * bad selector.
 */
function cloudModelUnconfirmedNote(
  profileName: string,
  llmProfileRef: string,
  reason: string,
) {
  return `Saved agent profile "${profileName}" exists on the cloud backend, but its model was not confirmed here: ${reason}. Organization-only LLM profiles are not listed on the per-user route while this conversation is local. The child still launches with this profile id and the cloud backend resolves it.`;
}

function resolvedOpenHandsProfile(
  profile: { id: string; name: string },
  model: string | null,
  llmProfileRef: string,
  secretRefsRestricted: boolean,
  modelNote?: string,
): ChildAgentProfileResolution {
  return {
    ok: true,
    profile: {
      id: profile.id,
      name: profile.name,
      agentKind: "openhands",
      model,
      llmProfileRef,
      secretRefsRestricted,
      ...(modelNote ? { modelNote } : {}),
    },
  };
}

/**
 * Resolve `agent_profile` against the target backend before any child exists.
 *
 * Accepts a saved agent profile's name or stable id. An unknown name, a
 * profile that cannot be read, or a local OpenHands profile whose LLM
 * profile is missing fails here so the caller never starts a conversation.
 * A cloud profile that exists still resolves when its model cannot be
 * confirmed from the LLM profiles visible to this session.
 */
export async function resolveChildAgentProfile(options: {
  selector: string;
  target: ChildConversationTarget;
  cloudBackend?: Backend | null;
}): Promise<ChildAgentProfileResolution> {
  const selector = options.selector.trim();
  const { target } = options;
  const cloudBackend = options.cloudBackend ?? null;
  const where = target === "cloud" ? "cloud" : "local";

  let listed;
  try {
    listed = await listProfiles(target, cloudBackend);
  } catch (error) {
    return rejected(
      `Could not read saved agent profiles on the ${where} backend (${errorDetail(error)}).`,
      `Nothing was launched, so ${JSON.stringify(selector)} was not applied. Retry when agent profiles are reachable, or omit \`agent_profile\` to keep the default settings path.`,
    );
  }

  const summary = matchProfile(listed.profiles ?? [], selector);
  if (!summary) {
    return rejected(
      `No saved agent profile matches ${JSON.stringify(selector)} on the ${where} backend.`,
      `Nothing was launched. Pass \`agent_profile\` as one of: ${savedProfileList(listed.profiles ?? [])}. Or omit \`agent_profile\` to keep the default settings path.`,
    );
  }

  let detail;
  try {
    detail = await readProfile(target, cloudBackend, summary.name);
  } catch (error) {
    return rejected(
      `Could not read agent profile "${summary.name}" on the ${where} backend (${errorDetail(error)}).`,
      `Nothing was launched. Retry when the profile can be read, choose another saved profile, or omit \`agent_profile\`.`,
    );
  }

  const profile = detail.profile;
  const secretRefsRestricted = Array.isArray(profile.secret_refs);
  const profileId = profile.id;
  if (!profileId) {
    return rejected(
      `Agent profile "${summary.name}" has no id on the ${where} backend.`,
      "Nothing was launched. Choose another saved agent profile, or omit `agent_profile`.",
    );
  }

  if (profile.agent_kind === "acp") {
    return {
      ok: true,
      profile: {
        id: profileId,
        name: profile.name,
        agentKind: "acp",
        model: profile.acp_model ?? null,
        llmProfileRef: null,
        secretRefsRestricted,
      },
    };
  }

  const llmProfileRef = profile.llm_profile_ref?.trim() ?? "";
  if (!llmProfileRef) {
    return rejected(
      `Agent profile "${profile.name}" has no LLM profile.`,
      "Nothing was launched. An OpenHands agent profile must reference a saved LLM profile before it can launch a child. Choose another profile, or omit `agent_profile`.",
    );
  }

  let model: string | null | undefined;
  try {
    model = await openHandsModel(target, cloudBackend, llmProfileRef);
  } catch (error) {
    if (target === "cloud") {
      return resolvedOpenHandsProfile(
        { id: profileId, name: profile.name },
        null,
        llmProfileRef,
        secretRefsRestricted,
        cloudModelUnconfirmedNote(
          profile.name,
          llmProfileRef,
          `LLM profiles could not be read (${errorDetail(error)})`,
        ),
      );
    }
    return rejected(
      `Could not confirm LLM profile "${llmProfileRef}" for agent profile "${profile.name}" on the ${where} backend (${errorDetail(error)}).`,
      "Nothing was launched. Retry when LLM profiles are reachable, or omit `agent_profile` to keep the default settings path.",
    );
  }

  if (model === undefined) {
    if (target === "cloud") {
      return resolvedOpenHandsProfile(
        { id: profileId, name: profile.name },
        null,
        llmProfileRef,
        secretRefsRestricted,
        cloudModelUnconfirmedNote(
          profile.name,
          llmProfileRef,
          `LLM profile "${llmProfileRef}" is not on this account's visible LLM profile list`,
        ),
      );
    }
    return rejected(
      `Agent profile "${profile.name}" references LLM profile "${llmProfileRef}", which is not saved on the ${where} backend.`,
      "Nothing was launched. Point the agent profile at a saved LLM profile, choose a different agent profile, or omit `agent_profile`.",
    );
  }

  return resolvedOpenHandsProfile(
    { id: profileId, name: profile.name },
    model,
    llmProfileRef,
    secretRefsRestricted,
  );
}
