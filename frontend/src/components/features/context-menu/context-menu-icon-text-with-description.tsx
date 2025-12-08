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
}

export function ContextMenuIconTextWithDescription({
  icon,
  title,
  description,
  className,
  iconClassName,
}: ContextMenuIconTextWithDescriptionProps) {
  return (
    <div
      className={cn(
        "flex flex-col gap-1 justify-center hover:bg-[#5C5D62] rounded p-2",
        className,
      )}
    >
      <ContextMenuIconText
        icon={icon}
        text={title}
        className="px-0"
        iconClassName={iconClassName}
      />
      <Typography.Text className="text-[#A3A3A3] text-[10px] font-normal whitespace-pre-wrap break-words">
        {description}
      </Typography.Text>
    </div>
  );
}
