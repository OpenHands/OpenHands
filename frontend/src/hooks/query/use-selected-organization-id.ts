import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useRevalidator } from "react-router";

export const SELECTED_ORGANIZATION_QUERY_KEY = "selected_organization";

export function useSelectedOrgId() {
  const queryClient = useQueryClient();
  const revalidator = useRevalidator();

  return useQuery({
    queryKey: [SELECTED_ORGANIZATION_QUERY_KEY],
    initialData: null as string | null,
    queryFn: () => {
      const storedOrgId = queryClient.getQueryData<string>([
        SELECTED_ORGANIZATION_QUERY_KEY,
      ]);
      // Revalidate route clientLoader to ensure the latest orgId is used.
      // This is useful for redirecting the user away from admin-only org pages.
      revalidator.revalidate();
      return storedOrgId || null; // Return null if no org ID is set
    },
  });
}
