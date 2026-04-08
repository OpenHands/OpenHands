import { OpenHandsObservation } from "#/types/core/observations";
import { isErrorObservation } from "#/types/core/guards";
import { ErrorMessage } from "../error-message";

interface ErrorEventMessageProps {
  event: OpenHandsObservation;
}

export function ErrorEventMessage({ event }: ErrorEventMessageProps) {
  if (!isErrorObservation(event)) {
    return null;
  }

  return (
    <ErrorMessage
      errorId={event.extras.error_id}
      defaultMessage={event.message}
    />
  );
}
