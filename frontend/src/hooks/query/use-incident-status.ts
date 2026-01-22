import { useQuery } from "@tanstack/react-query";
import type { IncidentStatusResponse } from "#/api/option-service/incident.types";
import { openHands } from "#/api/open-hands-axios";

interface UseIncidentStatusOptions {
  enabled?: boolean;
}

export const useIncidentStatus = (options?: UseIncidentStatusOptions) =>
  useQuery<IncidentStatusResponse>({
    queryKey: ["incident-status"],
    queryFn: async () => {
      const { data } = await openHands.get("/api/v1/status");
      return data;
    },
    staleTime: 1000 * 60,
    gcTime: 1000 * 60 * 5,
    refetchInterval: 1000 * 60 * 2,
    enabled: options?.enabled ?? true,
  });
