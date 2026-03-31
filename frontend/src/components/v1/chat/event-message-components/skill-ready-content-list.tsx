import React from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown, ChevronRight } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { SkillReadyItem } from "../event-content-helpers/create-skill-ready-event";
import { SkillItemExpanded } from "./skill-item-expanded";

interface SkillReadyContentListProps {
  items: SkillReadyItem[];
}

export function SkillReadyContentList({ items }: SkillReadyContentListProps) {
  const { t } = useTranslation();
  const [expandedSkills, setExpandedSkills] = React.useState<
    Record<string, boolean>
  >({});

  const toggleSkill = (name: string) => {
    setExpandedSkills((prev) => ({ ...prev, [name]: !prev[name] }));
  };

  return (
    <div className="flex flex-col gap-1 mt-1">
      <Typography.Text className="font-bold text-neutral-200 text-sm px-2 py-1">
        {t(I18nKey.SKILLS$TRIGGERED_SKILL_KNOWLEDGE)}
      </Typography.Text>
      {items.map((item) => {
        const isExpanded = expandedSkills[item.name] || false;

        return (
          <div
            key={item.name}
            className="border border-neutral-700 rounded-md overflow-hidden"
          >
            <button
              type="button"
              onClick={() => toggleSkill(item.name)}
              className="w-full py-1.5 px-2 text-left flex items-center gap-2 hover:bg-neutral-700 transition-colors cursor-pointer"
            >
              <Typography.Text className="text-neutral-300">
                {isExpanded ? (
                  <ChevronDown size={14} />
                ) : (
                  <ChevronRight size={14} />
                )}
              </Typography.Text>
              <Typography.Text className="font-semibold text-neutral-200 text-sm">
                {item.name}
              </Typography.Text>
            </button>

            {isExpanded && item.content && (
              <>
                <hr className="border-neutral-700" />
                <SkillItemExpanded content={item.content} />
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}
