import { useQuery } from "@tanstack/react-query";
import { getAuthScopes } from "#/api/auth-scopes";

export const useAuthScopes = () =>
  useQuery({
    queryKey: ["auth_scopes"],
    queryFn: getAuthScopes,
  });
