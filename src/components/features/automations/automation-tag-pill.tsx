import type { ReactNode } from "react";
import { Tag } from "lucide-react";
import { cn } from "#/utils/utils";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";

interface AutomationTagPillProps {
  tag: string;
  className?: string;
  /** Trailing content, e.g. the editor's remove button. */
  children?: ReactNode;
}

/** One automation tag, in the same pill chrome as the card metadata. */
export function AutomationTagPill({
  tag,
  className,
  children,
}: AutomationTagPillProps) {
  return (
    <span
      data-testid="automation-tag"
      className={cn(extensionModuleCardPillClassName, "gap-1", className)}
    >
      <Tag className="size-3 shrink-0" aria-hidden="true" />
      <span className="truncate" title={tag}>
        {tag}
      </span>
      {children}
    </span>
  );
}
