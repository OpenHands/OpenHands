import { useQuery } from "@tanstack/react-query";
import { organizationService } from "#/api/organization-service/organization-service.api";

export const useOrganizations = () =>
  useQuery({
    queryKey: ["organizations"],
    queryFn: organizationService.getOrganizations,
  });
