import { useQuery } from "@tanstack/react-query";
import { SuggestionsService } from "#/api/suggestions-service/suggestions-service.api";
import { groupSuggestedTasks } from "#/utils/group-suggested-tasks";
import { useShouldShowGitFeatures } from "../use-should-show-git-features";

export const useSuggestedTasks = () => {
  // Use the Git-specific hook since suggested tasks require a Git provider
  // to be configured (they come from Git repositories).
  const shouldShowGitFeatures = useShouldShowGitFeatures();

  return useQuery({
    queryKey: ["tasks"],
    queryFn: () => SuggestionsService.getSuggestedTasks(),
    select: groupSuggestedTasks,
    enabled: shouldShowGitFeatures,
  });
};
