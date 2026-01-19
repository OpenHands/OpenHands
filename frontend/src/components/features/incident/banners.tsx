import { IncidentBanner } from "./incident-banner";
import type {
  Incident,
  Maintenance,
} from "#/api/option-service/incident.types";
import { useIncidentStatus } from "#/hooks/query/use-incident-status";

function BannerList<T extends Incident | Maintenance>({
  items,
}: {
  items?: T[];
}) {
  if (!items || items.length === 0) {
    return null;
  }
  return (
    <>
      {items.map((item) => (
        <IncidentBanner key={item.id} incident={item} />
      ))}
    </>
  );
}

export function OngoingIncidentBanners() {
  const { data } = useIncidentStatus();
  return <BannerList items={data?.ongoing_incidents} />;
}
export function InProgressMaintenanceBanners() {
  const { data } = useIncidentStatus();
  return <BannerList items={data?.in_progress_maintenances} />;
}
export function ScheduledMaintenanceBanners() {
  const { data } = useIncidentStatus();
  return <BannerList items={data?.scheduled_maintenances} />;
}
