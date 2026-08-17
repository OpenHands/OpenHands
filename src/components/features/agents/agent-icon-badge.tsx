import RobotIcon from "#/icons/robot.svg?react";
import { cn } from "#/utils/utils";

interface AgentIconBadgeProps {
  agentName: string;
  className?: string;
}

export function AgentIconBadge({ agentName, className }: AgentIconBadgeProps) {
  return (
    <span
      aria-hidden="true"
      title={agentName}
      data-testid={`agent-icon-${agentName}`}
      className={cn(
        "inline-flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden",
        "rounded-lg border border-white/10 bg-surface-raised text-white",
        "shadow-[inset_0_1px_0_rgba(255,255,255,0.18)]",
        "[&>svg]:h-5 [&>svg]:w-5",
        className,
      )}
    >
      <RobotIcon />
    </span>
  );
}
