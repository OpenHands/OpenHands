import { useMutation, useQueryClient } from "@tanstack/react-query";
import { SELECTED_ORGANIZATION_QUERY_KEY } from "#/hooks/query/use-selected-organization-id";

export function useUpdateSelectedOrganizationId() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (newValue: string | null) => {
      queryClient.setQueryData([SELECTED_ORGANIZATION_QUERY_KEY], newValue);
      return newValue;
    },
    onMutate: async (newValue) => {
      await queryClient.cancelQueries({
        queryKey: [SELECTED_ORGANIZATION_QUERY_KEY],
      });

      // Snapshot the previous value
      const previousValue = queryClient.getQueryData([
        SELECTED_ORGANIZATION_QUERY_KEY,
      ]);
      queryClient.setQueryData([SELECTED_ORGANIZATION_QUERY_KEY], newValue);

      return { previousValue };
    },
    onError: (_, __, context) => {
      queryClient.setQueryData(
        [SELECTED_ORGANIZATION_QUERY_KEY],
        context?.previousValue,
      );
    },
    // Always refetch after error or success
    onSettled: () => {
      queryClient.invalidateQueries({
        queryKey: [SELECTED_ORGANIZATION_QUERY_KEY],
      });
    },
  });
}
