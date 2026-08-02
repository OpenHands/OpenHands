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
  const root = isRecord(response) ? response : {};
  const agent = isRecord(root.agent) ? root.agent : {};
  const agentContext = isRecord(agent.agent_context) ? agent.agent_context : {};

  const skills = Array.isArray(agentContext.skills)
    ? agentContext.skills
        .flatMap((skill) => {
          if (!isRecord(skill) || typeof skill.name !== "string") return [];
          return [
            {
              name: skill.name,
              description:
                typeof skill.description === "string"
                  ? skill.description
                  : null,
              source: typeof skill.source === "string" ? skill.source : null,
            },
          ];
        })
        .sort((left, right) => left.name.localeCompare(right.name))
    : [];

  const hookConfig = isRecord(root.hook_config) ? root.hook_config : {};
  const hooks = HOOK_TYPES.flatMap((hookType) => {
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
  });

  const mcpServerMap = getSdkMcpServerMap(agent.mcp_config);
  const mcps = mcpServerMap
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
    : [];

  return { skills, hooks, mcps };
}
