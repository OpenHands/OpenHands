import { useCallback } from "react";
import { useNavigation } from "#/context/navigation-context";
import { useConversationStore } from "#/stores/conversation-store";

export function useLaunchSkillInChat() {
  const { navigate } = useNavigation();
  const setMessageToSend = useConversationStore(
    (state) => state.setMessageToSend,
  );

  return useCallback(
    (message: string, onClose?: () => void) => {
      onClose?.();
      // Queued for the Home composer, which applies it once whether it is
      // already mounted or mounts after the navigation.
      setMessageToSend(message, "home");
      navigate("/conversations");
    },
    [navigate, setMessageToSend],
  );
}
