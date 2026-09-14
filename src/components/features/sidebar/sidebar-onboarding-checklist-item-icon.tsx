import { Plus } from "lucide-react";
import ClockIcon from "#/icons/clock.svg?react";
import KeyIcon from "#/icons/key.svg?react";
import ServerProcessIcon from "#/icons/server-process.svg?react";
import SlackIcon from "#/icons/slack.svg?react";
import { cn } from "#/utils/utils";
import type { SidebarOnboardingChecklistItemId } from "./sidebar-onboarding-checklist.constants";

const PREVIEW_ICON_SIZE = 18;

function CustomizeAgentIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={PREVIEW_ICON_SIZE}
      height={PREVIEW_ICON_SIZE}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M2.97 12.92A2 2 0 0 0 2 14.63v3.24a2 2 0 0 0 .97 1.71l3 1.8a2 2 0 0 0 2.06 0L12 19v-5.5l-5-3-4.03 2.42Z" />
      <path d="m7 16.5-4.74-2.85" />
      <path d="m7 16.5 5-3" />
      <path d="M7 16.5v5.17" />
      <path d="M12 13.5V19l3.97 2.38a2 2 0 0 0 2.06 0l3-1.8a2 2 0 0 0 .97-1.71v-3.24a2 2 0 0 0-.97-1.71L17 10.5l-5 3Z" />
      <path d="m17 16.5-5-3" />
      <path d="m17 16.5 4.74-2.85" />
      <path d="m17 16.5v5.17" />
      <path d="M7.97 4.42A2 2 0 0 0 7 6.13v4.37l5 3 5-3V6.13a2 2 0 0 0-.97-1.71l-3-1.8a2 2 0 0 0-2.06 0l-3 1.8Z" />
      <path d="M12 8 7.26 5.15" />
      <path d="m12 8 4.74-2.85" />
      <path d="M12 13.5V8" />
    </svg>
  );
}

interface SidebarOnboardingChecklistItemIconProps {
  id: SidebarOnboardingChecklistItemId;
  className?: string;
}

export function SidebarOnboardingChecklistItemIcon({
  id,
  className,
}: SidebarOnboardingChecklistItemIconProps) {
  const iconClassName = cn("shrink-0 text-white", className);
  const testId = `sidebar-onboarding-checklist-icon-${id}`;

  switch (id) {
    case "configure-llm":
      return (
        <span data-testid={testId} className="inline-flex shrink-0">
          <KeyIcon
            width={PREVIEW_ICON_SIZE}
            height={PREVIEW_ICON_SIZE}
            className={iconClassName}
            aria-hidden
          />
        </span>
      );
    case "start-conversation":
      return (
        <span data-testid={testId} className="inline-flex shrink-0">
          <Plus
            size={PREVIEW_ICON_SIZE}
            strokeWidth={2}
            className={iconClassName}
            aria-hidden
          />
        </span>
      );
    case "schedule-task":
      return (
        <span data-testid={testId} className="inline-flex shrink-0">
          <ClockIcon
            width={PREVIEW_ICON_SIZE}
            height={PREVIEW_ICON_SIZE}
            className={iconClassName}
            aria-hidden
          />
        </span>
      );
    case "customize-agent":
      return (
        <span
          data-testid={testId}
          className={cn("inline-flex shrink-0", iconClassName)}
        >
          <CustomizeAgentIcon />
        </span>
      );
    case "connect-mcp":
      return (
        <span data-testid={testId} className="inline-flex shrink-0">
          <ServerProcessIcon
            width={PREVIEW_ICON_SIZE}
            height={PREVIEW_ICON_SIZE}
            className={iconClassName}
            aria-hidden
          />
        </span>
      );
    case "join-slack":
      return (
        <span data-testid={testId} className="inline-flex shrink-0">
          <SlackIcon
            width={PREVIEW_ICON_SIZE}
            height={PREVIEW_ICON_SIZE}
            className={iconClassName}
            aria-hidden
          />
        </span>
      );
    default: {
      const unreachable: never = id;
      return unreachable;
    }
  }
}
