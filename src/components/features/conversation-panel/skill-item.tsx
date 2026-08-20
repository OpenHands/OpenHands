import { ChevronDown, ChevronRight } from "lucide-react";
import { Typography } from "#/ui/typography";
import { SkillTriggers } from "./skill-triggers";
import { SkillContent } from "./skill-content";
import { SkillInfo } from "#/types/settings";

interface SkillItemProps {
  skill: SkillInfo & { content?: string };
  isExpanded: boolean;
  onToggle: (agentName: string) => void;
}

const SKILL_TYPE_LABEL: Record<SkillInfo["type"], string> = {
  knowledge: "Knowledge",
  repo: "Repository",
  agentskills: "AgentSkills",
};

const SKILL_PILL_CLASS =
  "inline-flex shrink-0 items-center rounded-full px-2 py-0.5 text-[11px] font-medium leading-4 border border-[var(--oh-border)] bg-[var(--oh-surface)] text-tertiary-light";

export function SkillItem({ skill, isExpanded, onToggle }: SkillItemProps) {
  return (
    <div>
      <button
        type="button"
        onClick={() => onToggle(skill.name)}
        className="w-full py-3 px-3 text-left flex items-center justify-between hover:bg-tertiary transition-colors"
      >
        <div className="flex items-center">
          <Typography.Text className="font-bold text-content-2">
            {skill.name}
          </Typography.Text>
        </div>
        <div className="flex items-center gap-2">
          <span className={SKILL_PILL_CLASS}>
            {SKILL_TYPE_LABEL[skill.type]}
          </span>
          <Typography.Text className="text-[var(--oh-text-tertiary)]">
            {isExpanded ? (
              <ChevronDown size={18} />
            ) : (
              <ChevronRight size={18} />
            )}
          </Typography.Text>
        </div>
      </button>

      {isExpanded && (
        <div className="px-3 pb-3 pt-1 border-t border-[var(--oh-border)]">
          <SkillTriggers triggers={skill.triggers ?? []} />
          <SkillContent content={skill.content ?? ""} />
        </div>
      )}
    </div>
  );
}
