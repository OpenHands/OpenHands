export function SwitchSkeleton() {
  return (
    <div className="flex items-center gap-2">
      <div className="w-12 h-6 skeleton-round" />
      <div className="w-25 h-5 skeleton" />
    </div>
  );
}
