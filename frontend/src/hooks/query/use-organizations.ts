import { useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";

export const useOrganizations = () =>
  useQuery({
    queryKey: ["organizations"],
    queryFn: organizationService.getOrganizations,
    select: (data) =>
      // Sort organizations with personal workspace first, then alphabetically by name
      [...data].sort((a, b) => {
        const aIsPersonal = a.is_personal ?? false;
        const bIsPersonal = b.is_personal ?? false;
        if (aIsPersonal && !bIsPersonal) return -1;
        if (!aIsPersonal && bIsPersonal) return 1;
        return (a.name ?? "").localeCompare(b.name ?? "", undefined, {
          sensitivity: "base",
        });
      }),
  });
