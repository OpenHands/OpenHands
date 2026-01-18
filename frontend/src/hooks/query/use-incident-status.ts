import { useQuery } from "@tanstack/react-query";
import type { IncidentStatusResponse } from "#/api/option-service/incident.types";

interface UseIncidentStatusOptions {
  enabled?: boolean;
}

export const useIncidentStatus = (options?: UseIncidentStatusOptions) =>
  useQuery<IncidentStatusResponse>({
    queryKey: ["incident-status"],
    queryFn: async () => {
      const response = await fetch("/v1/status");
      if (!response.ok) {
        throw new Error("Failed to fetch incident status");
      }
      return response.json();
    },
    staleTime: 1000 * 60,
    gcTime: 1000 * 60 * 5,
    refetchInterval: 1000 * 60 * 2,
    enabled: options?.enabled ?? true,
  });
