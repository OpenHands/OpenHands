import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import CheckCircleIcon from "#/icons/u-check-circle.svg?react";
import { TaskItem } from "#/components/features/chat/task-tracking/task-item";
import { useTaskList } from "#/hooks/use-task-list";
import { cn } from "#/utils/utils";
import { ConversationTabEmptyState } from "#/components/features/conversation/conversation-tab-empty-state";

function TaskListTab() {
  const { t } = useTranslation("openhands");
  const { taskList } = useTaskList();

  if (taskList.length === 0) {
    return (
      <ConversationTabEmptyState icon={<CheckCircleIcon />}>
        {t(I18nKey.COMMON$NO_TASKS)}
      </ConversationTabEmptyState>
    );
  }

  return (
    <main className="h-full overflow-y-auto flex flex-col custom-scrollbar-always">
      {taskList.map((task) => (
        <div
          key={task.id}
          data-active={task.status === "in_progress" ? "true" : "false"}
          className={cn(
            "px-4 py-2",
            task.status === "in_progress" && "bg-[var(--oh-surface-raised)]",
          )}
        >
          <TaskItem task={task} />
        </div>
      ))}
    </main>
  );
}

export default TaskListTab;
