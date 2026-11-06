import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { SkillFacetRow } from "#/components/features/skills/skill-facet-row";
import type {
  MarketplaceFacetGroup,
  MarketplaceFacetGroupId,
} from "./marketplace-filter";

interface MarketplaceFacetRailProps {
  groups: MarketplaceFacetGroup[];
  onToggle: (groupId: MarketplaceFacetGroupId, value: string) => void;
  className?: string;
}

/** The Skills page's facet menu, with marketplace groups. */
export function MarketplaceFacetRail({
  groups,
  onToggle,
  className,
}: MarketplaceFacetRailProps) {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="marketplace-facet-rail"
      className={cn("flex min-w-0 flex-col gap-5", className)}
    >
      {groups.map((group) => {
        const titleId = `marketplace-facet-group-${group.id}-title`;
        return (
          <div
            key={group.id}
            data-testid={`marketplace-facet-group-${group.id}`}
            role="group"
            aria-labelledby={titleId}
            className="flex flex-col gap-1.5"
          >
            <span
              id={titleId}
              className="px-1 text-2xs font-semibold uppercase tracking-wider text-tertiary-alt"
            >
              {t(group.labelKey)}
            </span>
            {group.rows.map((row) => (
              <SkillFacetRow
                key={row.value}
                testId={`marketplace-facet-${group.id}-${row.value}`}
                {...(row.labelKey
                  ? { labelKey: row.labelKey }
                  : { label: row.label ?? row.value })}
                count={row.count}
                checked={row.checked}
                disabled={row.count === 0 && !row.checked}
                onToggle={() => onToggle(group.id, row.value)}
              />
            ))}
          </div>
        );
      })}
    </div>
  );
}
