export function InputSkeleton() {
  return (
    <div className="flex flex-col gap-2.5">
      <div className="w-17.5 h-5 skeleton" />
      <div className="w-full min-w-0 h-10 skeleton" />
    </div>
  );
}
