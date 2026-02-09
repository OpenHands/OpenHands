import { useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";

export const useOrganizations = () =>
  useQuery({
    queryKey: ["organizations"],
    queryFn: organizationService.getOrganizations,
    select: (data) =>
      // Sort organizations with personal workspace first
      [...data].sort((a, b) => {
        if (a.is_personal && !b.is_personal) return -1;
        if (!a.is_personal && b.is_personal) return 1;
        return 0;
      }),
  });
