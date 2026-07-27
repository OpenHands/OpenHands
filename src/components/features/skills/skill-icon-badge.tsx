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
        "rounded-lg border border-white/10 bg-surface-raised text-white",
        "shadow-[inset_0_1px_0_rgba(255,255,255,0.18)]",
        "[&>svg]:h-5 [&>svg]:w-5",
        className,
      )}
    >
      <SkillsIcon />
    </span>
  );
}
