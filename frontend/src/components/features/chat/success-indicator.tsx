import { FaClock } from "react-icons/fa";
import CheckCircle from "#/icons/check-circle-solid.svg?react";
import { ObservationResultStatus } from "./event-content-helpers/get-observation-result";

interface SuccessIndicatorProps {
  status: ObservationResultStatus;
}

export function SuccessIndicator({ status }: SuccessIndicatorProps) {
  return (
    <span className="flex-shrink-0">
      {status === "success" && (
        <CheckCircle
          data-testid="status-icon"
          className="h-4 w-4 ml-2 inline fill-success"
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
