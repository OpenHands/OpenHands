import { useLocation } from "react-router";
import { useTranslation } from "react-i18next";
import { useMemo } from "react";
import { useLocalStorage } from "@uidotdev/usehooks";
import { FaTriangleExclamation } from "react-icons/fa6";
import CloseIcon from "#/icons/close.svg?react";
import { cn } from "#/utils/utils";

interface MaintenanceBannerProps {
  startTime: string;
}

export function MaintenanceBanner({ startTime }: MaintenanceBannerProps) {
  const { t } = useTranslation();
  const [dismissedAt, setDismissedAt] = useLocalStorage<string | null>(
    "maintenance_banner_dismissed_at",
    null,
  );

  const { pathname } = useLocation();

  // Convert EST timestamp to user's local timezone
  const formatMaintenanceTime = (estTimeString: string): string => {
    try {
      // Parse the EST timestamp
      // If the string doesn't include timezone info, assume it's EST
      let dateToFormat: Date;

      if (
        estTimeString.includes("T") &&
        (estTimeString.includes("-05:00") ||
          estTimeString.includes("-04:00") ||
          estTimeString.includes("EST") ||
          estTimeString.includes("EDT"))
      ) {
        // Already has timezone info
        dateToFormat = new Date(estTimeString);
      } else {
        // Assume EST and convert to UTC for proper parsing
        // EST is UTC-5, EDT is UTC-4, but we'll assume EST for simplicity
        const estDate = new Date(estTimeString);
        if (Number.isNaN(estDate.getTime())) {
          throw new Error("Invalid date");
        }
        dateToFormat = estDate;
      }

      // Format to user's local timezone
      return dateToFormat.toLocaleString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        timeZoneName: "short",
      });
    } catch (error) {
      // Fallback to original string if parsing fails
      // eslint-disable-next-line no-console
      console.warn("Failed to parse maintenance time:", error);
      return estTimeString;
    }
  };

  const localTime = formatMaintenanceTime(startTime);

  const isBannerVisible = useMemo(() => {
    const isValid = !Number.isNaN(new Date(startTime).getTime());
    if (!isValid) {
      return false;
    }
    return dismissedAt !== localTime;
  }, [dismissedAt, startTime]);

  if (!isBannerVisible) {
    return null;
  }

  return (
    <div
      data-testid="maintenance-banner"
      className={cn(
        "bg-primary text-[#0D0F11] p-4 rounded",
        "flex flex-row items-center justify-between m-1",
        pathname === "/" && "mt-3 mr-3",
      )}
    >
      <div className="flex items-center">
        <div className="flex-shrink-0">
          <FaTriangleExclamation className="text-white align-middle" />
        </div>
        <div className="ml-3">
          <p className="text-sm font-medium">
            {t("MAINTENANCE$SCHEDULED_MESSAGE", { time: localTime })}
          </p>
        </div>
      </div>

      <button
        type="button"
        data-testid="dismiss-button"
        onClick={() => setDismissedAt(localTime)}
        className={cn(
          "bg-[#0D0F11] rounded-full w-5 h-5 flex items-center justify-center cursor-pointer",
        )}
      >
        <CloseIcon />
      </button>
    </div>
  );
}
