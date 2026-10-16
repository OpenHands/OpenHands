import { FaClock } from "react-icons/fa";
import XCircle from "#/icons/x-circle-solid.svg?react";
import { ObservationResultStatus } from "#/components/conversation-events/chat/event-content-helpers/get-observation-result";

interface SuccessIndicatorProps {
  status: ObservationResultStatus;
}

export function SuccessIndicator({ status }: SuccessIndicatorProps) {
  return (
    <span className="flex-shrink-0">
      {status === "error" && (
        <XCircle
          data-testid="status-icon"
          className="h-4 w-4 ml-2 inline fill-red-400"
        />
      )}

      {status === "timeout" && (
        <FaClock
          data-testid="status-icon"
          className="h-4 w-4 ml-2 inline fill-yellow-500"
        />
      )}
    </span>
  );
}
