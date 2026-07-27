import { useIsMutating } from "@tanstack/react-query";
import { useNavigation } from "#/context/navigation-context";

export const useIsCreatingConversation = () => {
  const navigation = useNavigation();
  const numberOfPendingMutations = useIsMutating({
    mutationKey: ["create-conversation"],
  });

  const { isNavigating } = navigation;
  const hasPendingMutations = numberOfPendingMutations > 0;

  return hasPendingMutations || isNavigating;
};
