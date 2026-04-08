import { AgentStatus } from "#/components/features/controls/agent-status";
import { Tools } from "../../controls/tools";
import { useUnifiedPauseConversationSandbox } from "#/hooks/mutation/use-unified-stop-conversation";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useV1PauseConversation } from "#/hooks/mutation/use-v1-pause-conversation";
import { useV1ResumeConversation } from "#/hooks/mutation/use-v1-resume-conversation";
import { ChangeAgentButton } from "../change-agent-button";

interface ChatInputActionsProps {
  disabled: boolean;
}

export function ChatInputActions({ disabled }: ChatInputActionsProps) {
  const pauseConversationSandboxMutation = useUnifiedPauseConversationSandbox();
  const v1PauseConversationMutation = useV1PauseConversation();
  const v1ResumeConversationMutation = useV1ResumeConversation();
  const { conversationId } = useConversationId();

  const handlePauseAgent = () => {
    // V1: Pause the conversation (agent execution)
    v1PauseConversationMutation.mutate({ conversationId });
  };

  const handleResumeAgentClick = () => {
    // V1: Resume the conversation (agent execution)
    v1ResumeConversationMutation.mutate({ conversationId });
  };

  const isPausing =
    pauseConversationSandboxMutation.isPending ||
    v1PauseConversationMutation.isPending;

  return (
    <div className="w-full flex items-center justify-between">
      <div className="flex items-center gap-1">
        <div className="flex items-center gap-4">
          <Tools />
          <ChangeAgentButton />
        </div>
      </div>
      <AgentStatus
        className="ml-2 md:ml-3"
        handleStop={handlePauseAgent}
        handleResumeAgent={handleResumeAgentClick}
        disabled={disabled}
        isPausing={isPausing}
      />
    </div>
  );
}
