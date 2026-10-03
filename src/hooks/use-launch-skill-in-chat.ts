import { useCallback } from "react";
import { useNavigation } from "#/context/navigation-context";
import { queueHomePromptDraft } from "#/hooks/chat/use-draft-persistence";

export function useLaunchSkillInChat() {
  const { navigate } = useNavigation();

  return useCallback(
    (message: string, onClose?: () => void) => {
      queueHomePromptDraft(message);
      onClose?.();
      navigate("/conversations");
    },
    [navigate],
  );
}
