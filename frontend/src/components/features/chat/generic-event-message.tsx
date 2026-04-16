import React from "react";
import ArrowDown from "#/icons/angle-down-solid.svg?react";
import ArrowUp from "#/icons/angle-up-solid.svg?react";
import { SuccessIndicator } from "./success-indicator";
import { ObservationResultStatus } from "./event-content-helpers/get-observation-result";
import { MarkdownRenderer } from "../markdown/markdown-renderer";

interface GenericEventMessageProps {
  title: React.ReactNode;
  details: string | React.ReactNode;
  success?: ObservationResultStatus;
  initiallyExpanded?: boolean;
  /** Where to place the expand/collapse chevron relative to the title. */
  chevronPosition?: "before" | "after";
  /** Extra content rendered at the end of the title row (right side). */
  titleTrailing?: React.ReactNode;
}

export function GenericEventMessage({
  title,
  details,
  success,
  initiallyExpanded = false,
  chevronPosition = "after",
  titleTrailing,
}: GenericEventMessageProps) {
  const [showDetails, setShowDetails] = React.useState(initiallyExpanded);

  const chevron = details ? (
    <button
      type="button"
      onClick={() => setShowDetails((prev) => !prev)}
      className="cursor-pointer text-left"
      aria-label={showDetails ? "Collapse" : "Expand"}
    >
      {showDetails ? (
        <ArrowUp
          className={`h-4 w-4 inline fill-neutral-300 ${
            chevronPosition === "after" ? "ml-2" : "mr-2"
          }`}
        />
      ) : (
        <ArrowDown
          className={`h-4 w-4 inline fill-neutral-300 ${
            chevronPosition === "after" ? "ml-2" : "mr-2"
          }`}
        />
      )}
    </button>
  ) : null;

  return (
    <div className="flex flex-col gap-2 border-l-2 pl-2 my-2 py-2 border-neutral-300 text-sm w-full">
      <div className="flex items-center justify-between font-bold text-neutral-300">
        <div className="flex items-center">
          {chevronPosition === "before" && chevron}
          {title}
          {chevronPosition === "after" && chevron}
        </div>

        <div className="flex items-center">
          {titleTrailing}
          {success && <SuccessIndicator status={success} />}
        </div>
      </div>

      {showDetails &&
        (typeof details === "string" ? (
          <MarkdownRenderer>{details}</MarkdownRenderer>
        ) : (
          details
        ))}
    </div>
  );
}
