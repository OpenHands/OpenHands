import type { AgentInfo } from "#/types/settings";
import { cn } from "#/utils/utils";
import { AgentLevelBadge } from "./agent-level-badge";
import {
  SKILL_CARD_PILL_CLASS,
  type SkillCardPill,
} from "#/components/features/skills/skill-card-pill-row";

interface BuildAgentPillsOptions {
  testIdPrefix?: string;
}

function pillTestId(
  prefix: string | undefined,
  agentName: string,
  suffix: string,
) {
  if (prefix) {
    return `${prefix}-${agentName}-${suffix}`;
  }
  return undefined;
}

/**
 * Metadata pills shown on agent cards and in the detail modal: the scope
 * (level) badge, then the model when it is pinned (not `inherit`). The card
 * and the detail modal show the same set.
 */
export function buildAgentPills(
  agent: AgentInfo,
  options: BuildAgentPillsOptions = {},
): SkillCardPill[] {
  const { testIdPrefix } = options;
  const pills: SkillCardPill[] = [];

  if (agent.level) {
    pills.push({
      id: `level-${agent.level}`,
      node: <AgentLevelBadge level={agent.level} />,
    });
  }

  const model = agent.model?.trim();
  if (model && model !== "inherit") {
    pills.push({
      id: `model-${model}`,
      node: (
        <span
          data-testid={
            pillTestId(testIdPrefix, agent.name, "model") ??
            `agent-model-${agent.name}`
          }
          className={cn(SKILL_CARD_PILL_CLASS, "font-mono")}
        >
          {model}
        </span>
      ),
    });
  }

  return pills;
}
