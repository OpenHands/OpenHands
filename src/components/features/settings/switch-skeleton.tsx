export function SwitchSkeleton() {
  return (
    <div className="flex items-center gap-2">
      <div className="w-[48px] h-[24px] skeleton-round" />
      <div className="w-[100px] h-[20px] skeleton" />
    </div>
  );
}
