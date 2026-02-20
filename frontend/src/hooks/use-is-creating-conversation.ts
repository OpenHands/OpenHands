import { useIsMutating } from "@tanstack/react-query";
import { useNavigation } from "react-router";

export const useIsCreatingConversation = () => {
  const navigation = useNavigation();
  const numberOfPendingCreateMutations = useIsMutating({
    mutationKey: ["create-conversation"],
  });
  const numberOfPendingConnectMutations = useIsMutating({
    mutationKey: ["connect-to-environment"],
  });

  const isNavigating = Boolean(navigation.location);
  const hasPendingMutations =
    numberOfPendingCreateMutations > 0 || numberOfPendingConnectMutations > 0;

  return hasPendingMutations || isNavigating;
};
