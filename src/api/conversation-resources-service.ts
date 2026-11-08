import {
  ConversationClient,
  SkillsClient,
} from "@openhands/typescript-client/clients";
import type { ConversationInfo } from "@openhands/typescript-client";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import type { ResolvedActiveBackend } from "#/api/backend-registry/types";
import type { ConversationRuntimeContext } from "#/api/conversation-file-upload.api";
import type { SkillInfo } from "#/types/settings";
import { assertActiveEnvironment } from "#/utils/assert-active-environment";
import { parseMcpConfig } from "#/utils/mcp-config";
import { flattenMcpConfig } from "#/utils/mcp-installed-servers";
import { getMcpServerHealthKey } from "#/utils/mcp-server-health-key";
import { resolveEffectiveAcpModel } from "#/constants/acp-providers";

export interface ConversationResources {
  agentKind: "openhands" | "acp";
  model: string | null;
  profile: {
    id: string;
    name: string | null;
    launchedRevision: number;
    currentRevision: number | null;
  } | null;
  skills: SkillInfo[] | null;
  mcpServers: { name: string; healthKey: string; type: string }[] | null;
  secretNames: string[] | null;
  condenser: {
    kind: string;
    maxSize?: number;
    keepFirst?: number;
    maxTokens?: number;
  } | null;
  verification: {
    kind: string | null;
    mode?: string;
    model?: string;
    threshold?: number;
    maxIterations?: number;
  } | null;
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function text(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

/** Project the published SDK response; never return LLM keys, secret values or skill bodies. */
function projectResources(info: ConversationInfo): ConversationResources {
  const agent = info.agent;
  const context = record(agent.agent_context);
  const condenser = record(agent.condenser);
  const critic = record(agent.critic);
  const refinement = record(critic?.iterative_refinement);
  const secretSources = record(record(info.secret_registry)?.secret_sources);
  const stamp = info.launched_agent_profile;
  const isAcp = agent.kind === "ACPAgent";
  const skills = context?.skills;

  return {
    agentKind: isAcp ? "acp" : "openhands",
    model: isAcp
      ? resolveEffectiveAcpModel({
          runtimeName: text(info.current_model_name),
          runtimeId: text(info.current_model_id),
          configured: text(agent.acp_model),
          sdkLlm: text(agent.llm?.model),
        })
      : text(agent.llm?.model),
    profile: stamp
      ? {
          id: stamp.agent_profile_id,
          name: null,
          launchedRevision: stamp.revision,
          currentRevision: null,
        }
      : null,
    skills: Array.isArray(skills)
      ? skills.flatMap((value): SkillInfo[] => {
          const skill = record(value);
          const name = text(skill?.name);
          return name
            ? [{ name, type: "knowledge", source: text(skill?.source) }]
            : [];
        })
      : null,
    mcpServers:
      "mcp_config" in agent
        ? flattenMcpConfig(parseMcpConfig(agent.mcp_config)).map((server) => ({
            name: server.name ?? server.id,
            healthKey: getMcpServerHealthKey(server),
            type: server.type,
          }))
        : null,
    secretNames: secretSources ? Object.keys(secretSources).sort() : null,
    condenser:
      condenser && text(condenser.kind)
        ? {
            kind: text(condenser.kind)!,
            ...(typeof condenser.max_size === "number"
              ? { maxSize: condenser.max_size }
              : {}),
            ...(typeof condenser.keep_first === "number"
              ? { keepFirst: condenser.keep_first }
              : {}),
            ...(typeof condenser.max_tokens === "number"
              ? { maxTokens: condenser.max_tokens }
              : {}),
          }
        : null,
    verification:
      "critic" in agent
        ? {
            kind: text(critic?.kind),
            ...(typeof critic?.mode === "string" ? { mode: critic.mode } : {}),
            ...(typeof critic?.model_name === "string"
              ? { model: critic.model_name }
              : {}),
            ...(typeof refinement?.success_threshold === "number"
              ? { threshold: refinement.success_threshold }
              : {}),
            ...(typeof refinement?.max_iterations === "number"
              ? { maxIterations: refinement.max_iterations }
              : {}),
          }
        : null,
  };
}

class ConversationResourcesService {
  static async load(
    conversationId: string,
    environment: ResolvedActiveBackend,
    runtime?: ConversationRuntimeContext,
  ): Promise<ConversationResources> {
    if (
      environment.backend.kind === "cloud" &&
      (!runtime?.conversationUrl || !runtime.sessionApiKey)
    ) {
      throw new Error("Conversation runtime is unavailable");
    }
    assertActiveEnvironment(environment);
    const options = getAgentServerClientOptions({
      conversationId,
      ...(environment.backend.kind === "cloud"
        ? runtime
        : {
            host: environment.backend.host,
            apiKey: environment.backend.apiKey,
          }),
    });
    const info = await new ConversationClient(options).getConversation(
      conversationId,
    );
    assertActiveEnvironment(environment);
    const resources = projectResources(info);

    // Project skills are discovered lazily when the agent uses its workspace;
    // unlike explicit/user skills they are not serialized in agent_context.skills.
    const context = record(info.agent.agent_context);
    if (
      environment.backend.kind === "local" &&
      resources.agentKind === "openhands" &&
      context?.load_project_skills === true
    ) {
      const workspace = record(info.workspace);
      if (
        workspace?.kind !== "LocalWorkspace" ||
        !text(workspace.working_dir)
      ) {
        resources.skills = null;
      } else {
        try {
          const { skills } = await new SkillsClient(options).getSkills({
            load_public: false,
            load_user: false,
            load_project: true,
            load_org: false,
            project_dir: text(workspace.working_dir),
          });
          assertActiveEnvironment(environment);
          const disabled = new Set(
            Array.isArray(context.disabled_skills)
              ? context.disabled_skills
              : [],
          );
          const byName = new Map(
            (resources.skills ?? []).map((skill) => [skill.name, skill]),
          );
          for (const skill of skills) {
            if (!disabled.has(skill.name))
              byName.set(skill.name, {
                name: skill.name,
                type: skill.type,
                source: skill.source ?? null,
              });
          }
          resources.skills = [...byName.values()];
        } catch {
          assertActiveEnvironment(environment);
          resources.skills = null;
        }
      }
    }

    if (resources.profile) {
      // Profile identity/revision only: current defaults are not the old conversation's configuration.
      try {
        const { profiles } = await AgentProfilesService.listProfiles();
        assertActiveEnvironment(environment);
        const profile = profiles.find(
          (candidate) => candidate.id === resources.profile?.id,
        );
        if (profile) {
          resources.profile.name = profile.name;
          resources.profile.currentRevision = profile.revision;
        }
      } catch {
        // Deleted/inaccessible profiles do not hide the runtime snapshot or launch stamp.
        assertActiveEnvironment(environment);
      }
    }
    return resources;
  }
}

export default ConversationResourcesService;
