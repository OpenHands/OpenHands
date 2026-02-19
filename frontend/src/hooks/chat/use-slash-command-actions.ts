import { useCallback } from "react";
import { useNavigate, useParams } from "react-router";
import { SlashCommand } from "#/config/slash-commands";
import { useClearConversation } from "#/hooks/mutation/use-clear-conversation";

export interface UseSlashCommandActionsReturn {
  executeCommand: (command: SlashCommand) => boolean;
}

export function useSlashCommandActions(): UseSlashCommandActionsReturn {
  const navigate = useNavigate();
  const { conversationId } = useParams<{ conversationId: string }>();
  const clearConversation = useClearConversation();

  const executeCommand = useCallback(
    (command: SlashCommand): boolean => {
      switch (command.action) {
        case "clear":
          // Clear conversation events on backend and local UI
          if (conversationId) {
            clearConversation.mutate({ conversationId });
          }
          return true;

        case "settings":
          navigate("/settings");
          return true;

        case "model":
          // Navigate to settings (model configuration is there)
          navigate("/settings");
          return true;

        default:
          return false;
      }
    },
    [navigate, conversationId, clearConversation],
  );

  return {
    executeCommand,
  };
}
