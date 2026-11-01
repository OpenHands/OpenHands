import SkillsIcon from "#/icons/skills.svg?react";
import { cn } from "#/utils/utils";

interface SkillIconBadgeProps {
  skillName: string;
  className?: string;
}

export function SkillIconBadge({ skillName, className }: SkillIconBadgeProps) {
  return (
    <span
      aria-hidden="true"
      title={skillName}
      data-testid={`skill-icon-${skillName}`}
      className={cn(
        "inline-flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden",
        "rounded-lg border border-contrast/10 bg-surface-raised text-contrast",
        "shadow-[inset_0_1px_0_color-mix(in_srgb,var(--oh-contrast)_18%,transparent)]",
        "[&>svg]:h-5 [&>svg]:w-5",
        className,
      )}
    >
      <SkillsIcon />
    </span>
  );
}
