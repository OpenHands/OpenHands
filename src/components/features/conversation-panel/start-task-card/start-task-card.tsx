import { useTranslation } from "react-i18next";
import type { AppConversationStartTask } from "#/api/conversation-service/agent-server-conversation-service.types";
import { I18nKey } from "#/i18n/declaration";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { ConversationCard } from "../conversation-card/conversation-card";

interface StartTaskCardProps {
  task: AppConversationStartTask;
  onClick?: () => void;
  initialMessage?: string;
  isActive?: boolean;
  showRepositoryMetadata?: boolean;
}

export function StartTaskCard({
  task,
  onClick,
  initialMessage,
  isActive,
  showRepositoryMetadata = false,
}: StartTaskCardProps) {
  const { t } = useTranslation("openhands");
  const prompt =
    initialMessage ||
    task.request.initial_message?.content
      .filter((content) => content.type === "text")
      .map((content) => content.text)
      .join(" ");
  const title =
    task.request.title?.trim() ||
    prompt?.trim() ||
    t(I18nKey.CONVERSATION$UNTITLED);

  const selectedRepository = task.request.selected_repository
    ? {
        selected_repository: task.request.selected_repository,
        selected_branch: task.request.selected_branch || null,
        git_provider: task.request.git_provider || null,
      }
    : null;

  return (
    <ConversationCard
      title={title}
      onClick={onClick}
      selectedRepository={selectedRepository}
      createdAt={task.created_at}
      lastUpdatedAt={task.updated_at}
      executionStatus={task.status === "ERROR" ? ExecutionStatus.ERROR : null}
      isActive={isActive}
      showRepositoryMetadata={showRepositoryMetadata}
    />
  );
}
