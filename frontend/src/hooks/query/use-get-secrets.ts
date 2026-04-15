import { useQuery } from "@tanstack/react-query";
import { SecretsService } from "#/api/secrets-service";
import { useConfig } from "./use-config";
import { useIsAuthed } from "#/hooks/query/use-is-authed";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";

export const useGetSecrets = () => {
  const { data: config } = useConfig();
  const { data: isAuthed } = useIsAuthed();
  const { organizationId } = useSelectedOrganizationId();

  const isOss = config?.app_mode === "oss";

  return useQuery({
    queryKey: ["secrets", organizationId],
    queryFn: SecretsService.getSecrets,
    enabled: isOss || (isAuthed && !!organizationId),
  });
};
