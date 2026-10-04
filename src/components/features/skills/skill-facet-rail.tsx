import type { SkillCategoryId } from "@openhands/extensions/skills";
import type { LucideIcon } from "lucide-react";
import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { SKILL_CATEGORY_ICONS } from "#/utils/skill-category";
import { SkillFacetRow } from "./skill-facet-row";
import type { SkillFacetGroup, SkillFacetGroupId } from "./skill-filter";

interface SkillFacetRailProps {
  groups: SkillFacetGroup[];
  onToggle: (groupId: SkillFacetGroupId, value: string) => void;
  className?: string;
}

/** Only the category group carries an icon; the rest are plain rows. */
function iconFor(
  groupId: SkillFacetGroupId,
  value: string,
): LucideIcon | undefined {
  if (groupId !== "category") return undefined;
  return SKILL_CATEGORY_ICONS[value as SkillCategoryId];
}

export function SkillFacetRail({
  groups,
  onToggle,
  className,
}: SkillFacetRailProps) {
  const { t } = useTranslation("openhands");

  if (groups.length === 0) return null;

  return (
    <div
      data-testid="skill-facet-rail"
      className={cn("flex min-w-0 flex-col gap-5", className)}
    >
      {groups.map((group) => {
        const titleId = `skill-facet-group-${group.id}-title`;
        return (
          <div
            key={group.id}
            data-testid={`skill-facet-group-${group.id}`}
            role="group"
            aria-labelledby={titleId}
            className="flex flex-col gap-1.5"
          >
            <span
              id={titleId}
              className="px-1 text-[11px] font-semibold uppercase tracking-wider text-tertiary-alt"
            >
              {t(group.labelKey)}
            </span>
            {group.rows.map((row) => (
              <SkillFacetRow
                key={row.value}
                testId={`skill-facet-${group.id}-${row.value}`}
                labelKey={row.labelKey}
                count={row.count}
                checked={row.checked}
                disabled={row.disabled}
                icon={iconFor(group.id, row.value)}
                onToggle={() => onToggle(group.id, row.value)}
              />
            ))}
          </div>
        );
      })}
    </div>
  );
}
