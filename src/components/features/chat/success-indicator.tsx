import { FaClock } from "react-icons/fa";
import { useTranslation } from "react-i18next";
import XCircleIcon from "#/icons/x-circle.svg?react";
import { ObservationResultStatus } from "#/components/conversation-events/chat/event-content-helpers/get-observation-result";
import { I18nKey } from "#/i18n/declaration";

interface SuccessIndicatorProps {
  status: ObservationResultStatus;
}

export function SuccessIndicator({ status }: SuccessIndicatorProps) {
  const { t } = useTranslation("openhands");

  return (
    <span className="flex-shrink-0">
      {status === "timeout" && (
        <FaClock
          data-testid="status-icon"
          className="h-4 w-4 ml-2 inline fill-yellow-500"
        />
      )}
      {status === "rejected" && (
        <span
          data-testid="rejected-marker"
          className="text-xs text-danger font-medium ml-2 px-1.5 py-0.5 rounded bg-danger/10 border border-danger/20 inline-flex items-center gap-1"
        >
          <XCircleIcon
            data-testid="status-icon"
            className="h-3 w-3 inline fill-danger"
          />
          <span>{t(I18nKey.ACTION$REJECT)}</span>
        </span>
      )}
    </span>
  );
}
