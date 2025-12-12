import { useTranslation } from "react-i18next";
import type { V1AppConversationStartTask } from "#/api/conversation-service/v1-conversation-service.types";
import { cn } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";
import { StartTaskCardHeader } from "./start-task-card-header";
import { StartTaskCardFooter } from "./start-task-card-footer";

interface StartTaskCardProps {
  task: V1AppConversationStartTask;
  onClick?: () => void;
}

export function StartTaskCard({ task, onClick }: StartTaskCardProps) {
  const { t } = useTranslation();
  const title =
    task.request.title ||
    task.detail ||
    t(I18nKey.CONVERSATION$STARTING_CONVERSATION);

  const selectedRepository = task.request.selected_repository
    ? {
        selected_repository: task.request.selected_repository,
        selected_branch: task.request.selected_branch || null,
        git_provider: task.request.git_provider || null,
      }
    : null;

  return (
    <div
      data-testid="start-task-card"
      onClick={onClick}
      className={cn(
        "relative h-auto w-full p-3.5 border-b border-neutral-600 cursor-pointer",
        "hover:bg-[#454545]",
      )}
    >
      <div className="flex items-center justify-between w-full">
        <StartTaskCardHeader title={title} taskStatus={task.status} />
      </div>

      <StartTaskCardFooter
        selectedRepository={selectedRepository}
        createdAt={task.created_at}
        detail={task.detail}
      />
    </div>
  );
}
