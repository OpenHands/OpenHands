import { useCallback, useState } from "react";
import { CustomChatInput } from "./custom-chat-input";
import { useBtwInterceptor } from "#/hooks/chat/use-btw-interceptor";
import { useGoalInterceptor } from "#/hooks/chat/use-goal-interceptor";
import { useModelInterceptor } from "#/hooks/chat/use-model-interceptor";
import { useCliCommandInterceptor } from "#/hooks/chat/use-cli-command-interceptor";
import { useChatAttachmentUpload } from "#/hooks/chat/use-chat-attachment-upload";
import { AgentState } from "#/types/agent-state";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { GitControlBar } from "./git-control-bar";
import { useConversationStore } from "#/stores/conversation-store";
import { useAgentState } from "#/hooks/use-agent-state";
import { useSubConversationTaskPolling } from "#/hooks/query/use-sub-conversation-task-polling";
import { partitionImagesForUpload } from "#/components/features/chat/utils/chat-input.utils";
import { isTaskPolling } from "#/utils/utils";
import { ConfirmationPolicyModal } from "./confirmation-policy-modal";

interface InteractiveChatBoxProps {
  onSubmit: (message: string, images: File[], files: File[]) => void;
  disabled?: boolean;
  hasStartedConversation?: boolean;
}

export function InteractiveChatBox({
  onSubmit,
  disabled = false,
  hasStartedConversation,
}: InteractiveChatBoxProps) {
  const {
    images,
    files,
    imagesMarkedUploadAsFile,
    clearAllFiles,
    subConversationTaskId,
  } = useConversationStore();
  const { curAgentState } = useAgentState();
  const { data: conversation } = useActiveConversation();
  const { conversationId: routeConversationId } = useOptionalConversationId();
  const conversationId = routeConversationId ?? conversation?.id ?? null;
  const [isConfirmationPolicyModalOpen, setConfirmationPolicyModalOpen] =
    useState(false);
  const openConfirmationPolicyModal = useCallback(
    () => setConfirmationPolicyModalOpen(true),
    [],
  );
  const closeConfirmationPolicyModal = useCallback(
    () => setConfirmationPolicyModalOpen(false),
    [],
  );

  const { taskStatus: subConversationTaskStatus } =
    useSubConversationTaskPolling(
      subConversationTaskId,
      conversation?.id || null,
    );

  const { handleUpload } = useChatAttachmentUpload();

  const handleAfterGoal = useBtwInterceptor(conversationId, (message) => {
    const { imagesToEmbed, imagesAsFiles } = partitionImagesForUpload(
      images,
      imagesMarkedUploadAsFile,
    );
    onSubmit(message, imagesToEmbed, [...files, ...imagesAsFiles]);
    clearAllFiles();
  });
  const handleAfterModel = useGoalInterceptor(conversationId, handleAfterGoal);
  const handleAfterCliCommands = useModelInterceptor(
    conversationId,
    handleAfterModel,
  );
  const handleSubmit = useCliCommandInterceptor(
    conversationId,
    handleAfterCliCommands,
    { onOpenConfirmationPolicy: openConfirmationPolicyModal },
  );

  const handleSuggestionsClick = (suggestion: string) => {
    handleSubmit(suggestion);
  };

  const isDisabled =
    disabled ||
    curAgentState === AgentState.AWAITING_USER_CONFIRMATION ||
    isTaskPolling(subConversationTaskStatus);

  return (
    <>
      <div data-testid="interactive-chat-box">
        <CustomChatInput
          disabled={isDisabled}
          isNewConversationPending={disabled}
          hasStartedConversation={hasStartedConversation}
          onSubmit={handleSubmit}
          onFilesPaste={handleUpload}
        />
        <div className="mt-3 pb-3">
          <GitControlBar onSuggestionsClick={handleSuggestionsClick} />
        </div>
      </div>
      {isConfirmationPolicyModalOpen && conversationId && (
        <ConfirmationPolicyModal
          conversationId={conversationId}
          conversationUrl={conversation?.conversation_url}
          sessionApiKey={conversation?.session_api_key}
          onClose={closeConfirmationPolicyModal}
        />
      )}
    </>
  );
}
