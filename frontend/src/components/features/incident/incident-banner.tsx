import { useMemo } from "react";
import { useLocalStorage } from "@uidotdev/usehooks";
import { FaTriangleExclamation, FaXmark } from "react-icons/fa6";
import { useTranslation } from "react-i18next";
import type {
  Incident,
  Maintenance,
} from "#/api/option-service/incident.types";

interface IncidentBannerProps {
  incident: Incident | Maintenance;
}

export function IncidentBanner({ incident }: IncidentBannerProps) {
  const { t } = useTranslation();
  const [dismissedIds, setDismissedIncidents] = useLocalStorage<Record<
    string,
    boolean
  > | null>("dismissed_incidents", null);

  const isDismissed = useMemo(
    () => (dismissedIds ? dismissedIds[incident.id] || false : false),
    [dismissedIds, incident.id],
  );

  const bannerColor = useMemo(() => {
    const colorMap: Record<string, string> = {
      investigating: "bg-red-500",
      identified: "bg-orange-500",
      monitoring: "bg-yellow-500",
      resolved: "bg-green-500",
      maintenance_scheduled: "bg-blue-500",
      maintenance_in_progress: "bg-blue-500",
      maintenance_complete: "bg-green-500",
    };

    return colorMap[incident.status] ?? "bg-gray-500";
  }, [incident.status]);

  const handleDismiss = () => {
    const updated = { ...dismissedIds, [incident.id]: true };
    setDismissedIncidents(updated);
  };

  if (isDismissed) {
    return null;
  }

  return (
    <div
      className={`${bannerColor} text-white rounded m-1 p-4 flex items-center gap-4`}
    >
      <FaTriangleExclamation className="text-xl flex-shrink-0" />

      <div className="flex-1">
        <p className="font-semibold">{incident.name}</p>
        <p className="text-sm opacity-90">{incident.last_update_message}</p>
        <a
          href={incident.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs underline hover:opacity-80"
        >
          {t("incident.viewDetails")}
        </a>
      </div>

      <button
        type="button"
        onClick={handleDismiss}
        className="bg-white/20 hover:bg-white/30 rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0"
        aria-label={t("incident.dismiss")}
      >
        <FaXmark className="text-xs" />
      </button>
    </div>
  );
}
