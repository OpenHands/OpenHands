import { ChevronDown, ChevronRight } from "lucide-react";
import { Typography } from "#/ui/typography";
import { SkillTriggers } from "./skill-triggers";
import { SkillContent } from "./skill-content";
import { Skill } from "#/api/conversation-service/v1-conversation-service.types";

interface SkillItemProps {
  skill: Skill;
  isExpanded: boolean;
  onToggle: (agentName: string) => void;
}

export function SkillItem({ skill, isExpanded, onToggle }: SkillItemProps) {
  let skillTypeLabel: string;
  let skillTypeClassName = "bg-gray-800 text-gray-200";
  if (skill.type === "repo") {
    skillTypeLabel = "Repository";
    skillTypeClassName = "bg-blue-900 text-blue-200";
  } else if (skill.type === "knowledge") {
    skillTypeLabel = "Knowledge";
    skillTypeClassName = "bg-green-900 text-green-200";
  } else {
    skillTypeLabel = "AgentSkills";
  }

  return (
    <div className="rounded-md overflow-hidden">
      <button
        type="button"
        onClick={() => onToggle(skill.name)}
        className="w-full py-3 px-2 text-left flex items-center justify-between hover:bg-gray-700 transition-colors"
      >
        <div className="flex items-center">
          <Typography.Text className="font-bold text-gray-100">
            {skill.name}
          </Typography.Text>
        </div>
        <div className="flex items-center">
          <Typography.Text
            className={`px-2 py-1 text-xs rounded-full mr-2 ${skillTypeClassName}`}
          >
            {skillTypeLabel}
          </Typography.Text>
          <Typography.Text className="text-gray-300">
            {isExpanded ? (
              <ChevronDown size={18} />
            ) : (
              <ChevronRight size={18} />
            )}
          </Typography.Text>
        </div>
      </button>

      {isExpanded && (
        <div className="px-2 pb-3 pt-1">
          <SkillTriggers triggers={skill.triggers} />
          <SkillContent content={skill.content} />
        </div>
      )}
    </div>
  );
}
