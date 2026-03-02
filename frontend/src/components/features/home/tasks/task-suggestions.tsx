import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router";
import { TaskGroup } from "./task-group";
import { useSuggestedTasks } from "#/hooks/query/use-suggested-tasks";
import { TaskSuggestionsSkeleton } from "./task-suggestions-skeleton";
import { cn, getDisplayedTaskGroups, getTotalTaskCount } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";
import { GitRepository } from "#/types/git";
import { useConfig } from "#/hooks/query/use-config";
import { useUserProviders } from "#/hooks/use-user-providers";
import { Typography } from "#/ui/typography";

interface TaskSuggestionsProps {
  filterFor?: GitRepository | null;
}

export function TaskSuggestions({ filterFor }: TaskSuggestionsProps) {
  const { t } = useTranslation();
  const [isExpanded, setIsExpanded] = useState(false);

  const { data: config } = useConfig();
  const { data: tasks, isLoading } = useSuggestedTasks();
  const { providers } = useUserProviders();

  const isOSS = config?.app_mode === "oss";
  const hasNoProviders = isOSS && providers.length === 0;

  const suggestedTasks = filterFor
    ? tasks?.filter(
        (element) =>
          element.title === filterFor.full_name &&
          !!element.tasks.find(
            (task) => task.git_provider === filterFor.git_provider,
          ),
      )
    : tasks;

  const hasSuggestedTasks = suggestedTasks && suggestedTasks.length > 0;

  // Get the task groups to display based on expanded state
  const displayedTaskGroups = getDisplayedTaskGroups(
    suggestedTasks,
    isExpanded,
  );

  // Check if there are more individual tasks to show
  const hasMoreTasks = getTotalTaskCount(suggestedTasks) > 3;

  const handleToggle = () => {
    setIsExpanded((prev) => !prev);
  };

  return (
    <section
      data-testid="task-suggestions"
      className="flex flex-1 min-w-0 flex-col"
    >
      <div
        className={cn(
          "flex items-center gap-2",
          !hasSuggestedTasks && "mb-[14px]",
        )}
      >
        <h3 className="text-xs leading-4 text-white font-semibold py-[14px] pl-[14px]">
          {t(I18nKey.TASKS$SUGGESTED_TASKS)}
        </h3>
      </div>

      <div className="flex flex-col">
        {isLoading && (
          <div className="px-[14px]">
            <TaskSuggestionsSkeleton />
          </div>
        )}
        {!hasSuggestedTasks &&
          !isLoading &&
          (hasNoProviders ? (
            <div className="px-[14px] flex flex-col gap-3 pb-6 sm:pb-8">
              <Typography.Text className="text-xs leading-4 text-white font-medium">
                {t(I18nKey.TASKS$NO_GIT_PROVIDERS_TITLE)}
              </Typography.Text>

              <Typography.Text className="text-xs leading-4 text-[#C9C9C9] font-normal">
                {t(I18nKey.TASKS$NO_GIT_PROVIDERS_DESCRIPTION)}
              </Typography.Text>

              <Link
                to="/settings/integrations"
                className="w-fit hover:underline"
              >
                <Typography.Text className="text-xs leading-4 text-[#FAFAFA] font-normal">
                  {t(I18nKey.TASKS$NO_GIT_PROVIDERS_CTA)}
                </Typography.Text>
              </Link>
            </div>
          ) : (
            <Typography.Text className="text-xs leading-4 text-white font-medium px-[14px]">
              {t(I18nKey.TASKS$NO_TASKS_AVAILABLE)}
            </Typography.Text>
          ))}

        {!isLoading &&
          displayedTaskGroups &&
          displayedTaskGroups.length > 0 && (
            <div className="flex flex-col">
              <div className="transition-all duration-300 ease-in-out overflow-y-auto custom-scrollbar">
                <div className="flex flex-col">
                  {displayedTaskGroups.map((taskGroup, index) => (
                    <TaskGroup
                      key={index}
                      title={decodeURIComponent(taskGroup.title)}
                      tasks={taskGroup.tasks}
                    />
                  ))}
                </div>
              </div>
            </div>
          )}
      </div>

      {!isLoading && hasMoreTasks && (
        <div className="flex justify-start mt-6 mb-8 ml-[14px]">
          <button
            type="button"
            onClick={handleToggle}
            className="text-xs leading-4 text-[#FAFAFA] font-normal cursor-pointer hover:underline"
          >
            {isExpanded
              ? t(I18nKey.COMMON$VIEW_LESS)
              : t(I18nKey.COMMON$VIEW_MORE)}
          </button>
        </div>
      )}
    </section>
  );
}
