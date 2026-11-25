import React from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown, ChevronRight } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { SkillReadyItem } from "../event-content-helpers/create-skill-ready-event";
import { SkillItemExpanded } from "./skill-item-expanded";
import { useEventExpansionKey } from "#/components/features/chat/row-expansion-context";
import { usePersistentExpansion } from "#/stores/message-expansion-store";

interface SkillReadyContentListProps {
  items: SkillReadyItem[];
  /**
   * Translation key for the list header. Defaults to the "Triggered Skill
   * Knowledge:" label used by Skill Ready events; invoke-skill observations
   * pass "Invoked Skill Knowledge:" instead.
   */
  titleKey?: I18nKey;
}

/**
 * One expandable skill card. A component of its own so each card can key its
 * persisted expansion by the skill name (hooks cannot be called in a loop), so
 * a row remounting after virtualization restores exactly the cards the user had
 * opened.
 */
function SkillReadyItemRow({ item }: { item: SkillReadyItem }) {
  const expansionKey = useEventExpansionKey(`skill-${item.name}`);
  const [isExpanded, toggleExpanded] = usePersistentExpansion(
    expansionKey,
    false,
  );

  return (
    <div className="border border-border-subtle rounded-md overflow-hidden">
      <button
        type="button"
        onClick={() => toggleExpanded()}
        className="w-full py-1.5 px-2 text-left flex items-center gap-2 hover:bg-tertiary transition-colors cursor-pointer"
      >
        <Typography.Text className="text-text-tertiary">
          {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </Typography.Text>
        <Typography.Text className="font-normal text-foreground text-sm">
          {item.name}
        </Typography.Text>
      </button>

      {isExpanded && item.content && (
        <>
          <hr className="border-border-subtle" />
          <SkillItemExpanded content={item.content} />
        </>
      )}
    </div>
  );
}

export function SkillReadyContentList({
  items,
  titleKey = I18nKey.SKILLS$TRIGGERED_SKILL_KNOWLEDGE,
}: SkillReadyContentListProps) {
  const { t } = useTranslation("openhands");

  return (
    <div className="flex flex-col gap-1 mt-1">
      <Typography.Text className="font-bold text-foreground text-sm px-2 py-1">
        {t(titleKey)}
      </Typography.Text>
      {items.map((item) => (
        <SkillReadyItemRow key={item.name} item={item} />
      ))}
    </div>
  );
}
