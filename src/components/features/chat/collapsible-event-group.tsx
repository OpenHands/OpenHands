import React from "react";
import ArrowDown from "#/icons/angle-down-solid.svg?react";
import ArrowUp from "#/icons/angle-up-solid.svg?react";

interface Props {
  title: React.ReactNode;
  trailing?: React.ReactNode;
  expandLabel: string;
  collapseLabel: string;
  children: React.ReactNode;
  testId?: string;
}

export function CollapsibleEventGroup({
  title,
  trailing,
  expandLabel,
  collapseLabel,
  children,
  testId = "event-group",
}: Props) {
  const [expanded, setExpanded] = React.useState(false);
  const contentId = React.useId();
  const buttonId = `${contentId}-toggle`;
  const Chevron = expanded ? ArrowUp : ArrowDown;

  return (
    <div className="my-1 w-full py-1 text-sm" data-testid={testId}>
      <button
        id={buttonId}
        type="button"
        onClick={() => setExpanded((previous) => !previous)}
        aria-controls={contentId}
        aria-expanded={expanded}
        aria-label={expanded ? collapseLabel : expandLabel}
        data-testid={`${testId}-toggle`}
        className="w-full flex items-center justify-between gap-2 text-left cursor-pointer"
      >
        <span className="flex items-center gap-2 min-w-0 font-normal text-muted">
          <Chevron className="h-4 w-4 fill-muted flex-shrink-0" />
          <span className="truncate">{title}</span>
        </span>
        {trailing && (
          <span className="flex items-center flex-shrink-0 font-normal text-muted">
            {trailing}
          </span>
        )}
      </button>
      {expanded && (
        <div
          id={contentId}
          role="region"
          aria-labelledby={buttonId}
          className="mt-1.5 flex flex-col"
          data-testid={`${testId}-content`}
        >
          {children}
        </div>
      )}
    </div>
  );
}
