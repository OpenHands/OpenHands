import { cn } from "#/utils/utils";

function SkeletonBlock({ className }: { className: string }) {
  return (
    <div className={cn("animate-pulse rounded bg-surface-raised", className)} />
  );
}

export function GitSyncSkeleton() {
  return (
    <div className="flex flex-col gap-4" data-testid="git-sync-skeleton">
      <SkeletonBlock className="h-5 w-40" />
      <SkeletonBlock className="h-6 w-48" />
      <SkeletonBlock className="h-5 w-96" />
      <SkeletonBlock className="h-48 w-full rounded-2xl" />
      <SkeletonBlock className="h-96 w-full rounded-2xl" />
    </div>
  );
}
