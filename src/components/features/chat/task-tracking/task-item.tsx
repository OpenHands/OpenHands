import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import CircleIcon from "#/icons/u-circle.svg?react";
import CheckCircleIcon from "#/icons/u-check-circle.svg?react";
import CheckCircleHalfIcon from "#/icons/u-check-circle-half.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { Typography } from "#/ui/typography";

interface TaskItemProps {
  task: {
    id: string;
    title: string;
    status: "todo" | "in_progress" | "done";
    notes?: string;
  };
}

export function TaskItem({ task }: TaskItemProps) {
  const { t } = useTranslation("openhands");

  const icon = useMemo(() => {
    switch (task.status) {
      case "todo":
        return <CircleIcon className="w-4 h-4 text-[#ffffff]" />;
      case "in_progress":
        return <CheckCircleHalfIcon className="w-4 h-4 text-[#ffffff]" />;
      case "done":
        return <CheckCircleIcon className="w-4 h-4 text-[var(--oh-muted)]" />;
      default:
        return <CircleIcon className="w-4 h-4 text-[#ffffff]" />;
    }
  }, [task.status]);

  const isDoneStatus = task.status === "done";

  return (
    <div className="flex gap-2 items-center w-full" data-name="item">
      <div className="shrink-0">{icon}</div>
      <div className="flex flex-col items-start justify-center leading-[16px] text-nowrap whitespace-pre font-normal">
        <Typography.Text
          className={cn(
            "text-[12px]",
            isDoneStatus ? "text-[var(--oh-muted)]" : "text-white",
          )}
        >
          {task.title}
        </Typography.Text>
        {task.notes && (
          <Typography.Text className="text-[10px] text-[var(--oh-muted)]">
            {t(I18nKey.TASK_TRACKING_OBSERVATION$TASK_NOTES)}: {task.notes}
          </Typography.Text>
        )}
      </div>
    </div>
  );
}
