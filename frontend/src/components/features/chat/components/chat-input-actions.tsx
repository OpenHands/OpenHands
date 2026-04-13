import { useState } from "react";
import { AgentStatus } from "#/components/features/controls/agent-status";
import { Tools } from "../../controls/tools";
import { useUnifiedPauseConversationSandbox } from "#/hooks/mutation/use-unified-stop-conversation";
import { useConversationId } from "#/hooks/use-conversation-id";
import { useV1PauseConversation } from "#/hooks/mutation/use-v1-pause-conversation";
import { useV1ResumeConversation } from "#/hooks/mutation/use-v1-resume-conversation";
import { ChangeAgentButton } from "../change-agent-button";
import MoreIcon from "#/icons/more-horizontal.svg?react";
import { cn } from "#/utils/utils";

interface ChatInputActionsProps {
  disabled: boolean;
  isMobile?: boolean;
}

export function ChatInputActions({
  disabled,
  isMobile = false,
}: ChatInputActionsProps) {
  const [isExpanded, setIsExpanded] = useState(false);
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
    <div className="w-full">
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-1 items-center gap-4 min-w-0">
          <Tools />
          {(!isMobile || isExpanded) && <ChangeAgentButton />}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {isMobile && (
            <button
              type="button"
              onClick={() => setIsExpanded((value) => !value)}
              className={cn(
                "flex h-6 w-6 items-center justify-center rounded-full border border-[#4B505F] transition-opacity",
                "cursor-pointer hover:opacity-80",
              )}
              aria-expanded={isExpanded}
              aria-label="Toggle composer actions"
            >
              <MoreIcon width={16} height={16} color="#ffffff" />
            </button>
          )}

          {(!isMobile || isExpanded) && (
            <AgentStatus
              className="ml-0 md:ml-3"
              handleStop={handlePauseAgent}
              handleResumeAgent={handleResumeAgentClick}
              disabled={disabled}
              isPausing={isPausing}
            />
          )}
        </div>
      </div>
    </div>
  );
}
