import type { LoadedResources } from "#/types/slash-command";
import type { SkillInfo } from "#/types/settings";
import { getSdkMcpServerMap } from "#/utils/mcp-config";
import type { GetHooksResponse } from "#/api/conversation-service/agent-server-conversation-service.types";
import { getSkillDescription } from "#/utils/skill-description";

const HOOK_TYPES = [
  "pre_tool_use",
  "post_tool_use",
  "user_prompt_submit",
  "session_start",
  "session_end",
  "stop",
] as const;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function toLoadedSkillResources(
  skills: SkillInfo[],
): LoadedResources["skills"] {
  return skills
    .map((skill) => ({
      name: skill.name,
      description:
        skill.description ??
        (typeof skill.content === "string"
          ? getSkillDescription(skill.content)
          : null),
      source: skill.source ?? null,
    }))
    .sort((left, right) => left.name.localeCompare(right.name));
}

/** Convert the shared hooks response into the compact `/skills` display form. */
export function toLoadedHookResources(
  response: GetHooksResponse,
): NonNullable<LoadedResources["hooks"]> {
  return response.hooks.flatMap((event) => {
    const commands = event.matchers.flatMap((matcher) =>
      (matcher.hooks ?? []).flatMap((hook) =>
        hook.command ? [hook.command] : [],
      ),
    );

    return commands.length > 0
      ? [{ hookType: event.event_type, commands }]
      : [];
  });
}

/** Extract hooks and MCPs persisted with a conversation. */
export function parseLoadedResources(response: unknown): LoadedResources {
  if (!isRecord(response) || !isRecord(response.agent)) {
    throw new Error("Loaded resource data is unavailable in this runtime.");
  }
  const root = response;
  const agent = response.agent;
  const agentContext = isRecord(agent.agent_context)
    ? agent.agent_context
    : null;

  // `include_skills=true` promises this serialized field. If it is absent,
  // treating the response as an authoritative empty snapshot would hide a
  // stopped/older/incompatible runtime as "no resources".
  if (!agentContext || !Array.isArray(agentContext.skills)) {
    throw new Error("Loaded skill data is unavailable in this runtime.");
  }

  const skills = agentContext.skills
    .flatMap((skill) => {
      if (!isRecord(skill) || typeof skill.name !== "string") return [];
      return [
        {
          name: skill.name,
          description:
            typeof skill.description === "string" ? skill.description : null,
          source: typeof skill.source === "string" ? skill.source : null,
        },
      ];
    })
    .sort((left, right) => left.name.localeCompare(right.name));

  const hasSerializedHookConfig = Object.prototype.hasOwnProperty.call(
    root,
    "hook_config",
  );
  const hookConfig = isRecord(root.hook_config) ? root.hook_config : {};
  const hooks = hasSerializedHookConfig
    ? HOOK_TYPES.flatMap((hookType) => {
        const matchers = hookConfig[hookType];
        if (!Array.isArray(matchers)) return [];

        const commands = matchers.flatMap((matcher) => {
          if (!isRecord(matcher) || !Array.isArray(matcher.hooks)) return [];
          return matcher.hooks.flatMap((hook) =>
            isRecord(hook) && typeof hook.command === "string" && hook.command
              ? [hook.command]
              : [],
          );
        });

        return commands.length > 0 ? [{ hookType, commands }] : [];
      })
    : null;

  const hasSerializedMcpConfig = Object.prototype.hasOwnProperty.call(
    agent,
    "mcp_config",
  );
  const mcpServerMap = getSdkMcpServerMap(agent.mcp_config);
  const mcps = hasSerializedMcpConfig
    ? mcpServerMap
      ? Object.entries(mcpServerMap).flatMap(([name, value]) => {
          if (!isRecord(value) || value.enabled === false) return [];
          const transport =
            typeof value.transport === "string"
              ? value.transport
              : typeof value.command === "string"
                ? "stdio"
                : typeof value.url === "string"
                  ? "http"
                  : null;
          return [{ name, transport }];
        })
      : []
    : null;

  return { skills, hooks, mcps };
}
