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
}

export function GenericEventMessage({
  title,
  details,
  success,
  initiallyExpanded = false,
}: GenericEventMessageProps) {
  const [showDetails, setShowDetails] = React.useState(initiallyExpanded);

  return (
    <div className="flex flex-col gap-2 border-l-2 pl-2 my-2 py-2 border-neutral-300 text-sm w-full">
      <div className="flex items-center justify-between font-bold text-neutral-300">
        <div>
          {title}
          {details && (
            <button
              type="button"
              onClick={() => setShowDetails((prev) => !prev)}
              className="cursor-pointer text-left"
            >
              {showDetails ? (
                <ArrowUp className="h-4 w-4 ml-2 inline fill-neutral-300" />
              ) : (
                <ArrowDown className="h-4 w-4 ml-2 inline fill-neutral-300" />
              )}
            </button>
          )}
        </div>

        {success && <SuccessIndicator status={success} />}
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
