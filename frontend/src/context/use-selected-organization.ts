import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRevalidator } from "react-router";

const key = "selected_organization" as const;

export const useSelectedOrganizationId = () => {
  const queryClient = useQueryClient();
  const revalidator = useRevalidator();

  const { data: orgId } = useQuery({
    queryKey: [key],
    initialData: null as string | null,
    queryFn: () => {
      const storedOrgId = queryClient.getQueryData<string>([key]);
      // Revalidate route clientLoader to ensure the latest orgId is used.
      // This is useful for redirecting the user away from admin-only org pages.
      revalidator.revalidate();
      return storedOrgId || null; // Return null if no org ID is set
    },
  });

  const updateState = useMutation({
    mutationFn: async (newValue: string | null) => {
      queryClient.setQueryData([key], newValue);
      return newValue;
    },
    onMutate: async (newValue) => {
      await queryClient.cancelQueries({ queryKey: [key] });

      // Snapshot the previous value
      const previousValue = queryClient.getQueryData([key]);
      queryClient.setQueryData([key], newValue);

      return { previousValue };
    },
    onError: (_, __, context) => {
      queryClient.setQueryData([key], context?.previousValue);
    },
    // Always refetch after error or success
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: [key] });
    },
  });

  return {
    orgId,
    setOrgId: (newValue: string | null) => updateState.mutate(newValue),
  };
};
