import { useLocation } from "react-router";
import { useTranslation } from "react-i18next";
import { useMemo } from "react";
import { useLocalStorage } from "@uidotdev/usehooks";
import { FaTriangleExclamation } from "react-icons/fa6";
import CloseIcon from "#/icons/close.svg?react";
import { cn } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";

interface AlertBannerProps {
  maintenanceStartTime?: string | null;
  faultyModels?: string[];
  errorMessage?: string | null;
  updatedAt: string;
}

export function AlertBanner({
  maintenanceStartTime,
  faultyModels,
  errorMessage,
  updatedAt,
}: AlertBannerProps) {
  const { t } = useTranslation();
  const [dismissedAt, setDismissedAt] = useLocalStorage<string | null>(
    "alert_banner_dismissed_at",
    null,
  );

  const { pathname } = useLocation();

  // Format ISO timestamp to user's local timezone
  const formatMaintenanceTime = (isoTimeString: string): string => {
    const date = new Date(isoTimeString);
    return date.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      timeZoneName: "short",
    });
  };

  const localTime = maintenanceStartTime
    ? formatMaintenanceTime(maintenanceStartTime)
    : null;

  const hasMaintenanceAlert = !!maintenanceStartTime;
  const hasFaultyModels = faultyModels && faultyModels.length > 0;
  const hasErrorMessage = errorMessage && errorMessage.trim().length > 0;

  const hasAnyAlert = hasMaintenanceAlert || hasFaultyModels || hasErrorMessage;

  const isBannerVisible = useMemo(() => {
    if (!hasAnyAlert) {
      return false;
    }
    return dismissedAt !== updatedAt;
  }, [dismissedAt, updatedAt, hasAnyAlert]);

  // Try to translate error message, fallback to raw message
  const translatedErrorMessage = useMemo(() => {
    if (!errorMessage) return null;

    // Check if the error message is a translation key (e.g., "ERROR$SOME_KEY")
    const translated = t(errorMessage as I18nKey);
    // If translation returns the same key, it means no translation exists
    if (translated === errorMessage) {
      return errorMessage;
    }
    return translated;
  }, [errorMessage, t]);

  if (!isBannerVisible) {
    return null;
  }

  const renderMessages = () => {
    const messages: React.ReactNode[] = [];

    if (hasMaintenanceAlert && localTime) {
      messages.push(
        <Typography.Paragraph key="maintenance" className="text-sm font-medium">
          {t(I18nKey.MAINTENANCE$SCHEDULED_MESSAGE, { time: localTime })}
        </Typography.Paragraph>,
      );
    }

    if (hasFaultyModels) {
      messages.push(
        <Typography.Paragraph
          key="faulty-models"
          className="text-sm font-medium"
        >
          {t(I18nKey.ALERT$FAULTY_MODELS_MESSAGE)} {faultyModels!.join(", ")}
        </Typography.Paragraph>,
      );
    }

    if (hasErrorMessage && translatedErrorMessage) {
      messages.push(
        <Typography.Paragraph
          key="error-message"
          className="text-sm font-medium"
        >
          {translatedErrorMessage}
        </Typography.Paragraph>,
      );
    }

    return messages;
  };

  return (
    <div
      data-testid="alert-banner"
      className={cn(
        "bg-[#0D0F11] border border-primary text-white p-4 rounded",
        "flex flex-row items-center justify-between m-1",
        pathname === "/" && "mt-3 mr-3",
      )}
    >
      <div className="flex items-center">
        <div className="flex-shrink-0">
          <FaTriangleExclamation className="text-primary align-middle" />
        </div>
        <div className="ml-3 flex flex-col gap-1">{renderMessages()}</div>
      </div>

      <button
        type="button"
        data-testid="dismiss-button"
        onClick={() => setDismissedAt(updatedAt)}
        className={cn(
          "bg-[#0D0F11] rounded-full w-5 h-5 flex items-center justify-center cursor-pointer",
        )}
      >
        <CloseIcon />
      </button>
    </div>
  );
}
