import React from "react";
import { ContextMenuIconText } from "./context-menu-icon-text";
import { Typography } from "#/ui/typography";
import { cn } from "#/utils/utils";

interface ContextMenuIconTextWithDescriptionProps {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  className?: string;
  iconClassName?: string;
  isActive?: boolean;
}

export function ContextMenuIconTextWithDescription({
  icon,
  title,
  description,
  className,
  iconClassName,
  isActive = false,
}: ContextMenuIconTextWithDescriptionProps) {
  return (
    <div
      className={cn(
        "flex min-w-0 w-full flex-col justify-center gap-1",
        className,
      )}
    >
      <ContextMenuIconText
        icon={icon}
        text={title}
        className="px-0"
        iconClassName={iconClassName}
        isActive={isActive}
      />
      <Typography.Text className="text-[var(--oh-muted)] text-[10px] font-normal whitespace-pre-wrap break-words">
        {description}
      </Typography.Text>
    </div>
  );
}
