import { useQuery } from "@tanstack/react-query";
import { SuggestionsService } from "#/api/suggestions-service/suggestions-service.api";
import { groupSuggestedTasks } from "#/utils/group-suggested-tasks";
import { useShouldShowUserFeatures } from "../use-should-show-user-features";

export const useSuggestedTasks = () => {
  const shouldShowUserFeatures = useShouldShowUserFeatures();

  return useQuery({
    queryKey: ["tasks"],
    queryFn: () => SuggestionsService.getSuggestedTasks(),
    select: groupSuggestedTasks,
    enabled: shouldShowUserFeatures,
  });
};
