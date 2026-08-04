import { useCallback } from "react";
import { useNavigation } from "#/context/navigation-context";
import {
  HOME_COMPOSER_KEY,
  useConversationStore,
} from "#/stores/conversation-store";

export function useLaunchSkillInChat() {
  const { navigate } = useNavigation();

  return useCallback(
    (message: string, onClose?: () => void) => {
      onClose?.();
      navigate("/conversations");
      window.setTimeout(() => {
        // Skills launch lands on the home launcher composer.
        useConversationStore
          .getState()
          .setMessageToSend(HOME_COMPOSER_KEY, message);
      }, 0);
    },
    [navigate],
  );
}
