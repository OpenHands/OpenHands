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
      setMessageToSend(message, null);
      navigate("/conversations");
    },
    [navigate, setMessageToSend],
  );
}
