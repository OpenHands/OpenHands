import { cn } from "#/utils/utils";

function TaskCardSkeleton() {
  return (
    <li className="py-3 flex items-center pr-6">
      <div className="h-3 w-8 skeleton !rounded-sm" />

      <div className="w-full pl-8">
        <div className="h-3 w-24 skeleton mb-2 !rounded-sm" />
        <div className={cn("h-3 skeleton w-3/4 !rounded-sm")} />
      </div>

      <div className="h-3 w-16 skeleton !rounded-sm" />
    </li>
  );
}

interface TaskGroupSkeletonProps {
  items?: number;
}

function TaskGroupSkeleton({ items = 3 }: TaskGroupSkeletonProps) {
  return (
    <div data-testid="task-group-skeleton">
      <div className="py-3 border-b border-[var(--oh-border-input)] pt-3.5">
        <div className="h-3 w-40 skeleton !rounded-sm" />
      </div>

      <ul>
        {Array.from({ length: items }).map((_, index) => (
          <TaskCardSkeleton key={index} />
        ))}
      </ul>

      <div className="w-15 h-3 skeleton !rounded-sm mb-4" />
    </div>
  );
}

export function TaskSuggestionsSkeleton() {
  return <TaskGroupSkeleton />;
}
