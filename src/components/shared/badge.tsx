import { cn } from "#/utils/utils";

export function BrandBadge({
  children,
  className,
  ...rest
}: React.HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={cn(
        "text-sm leading-4 text-black font-semibold tracking-tighter bg-primary p-1 rounded-full",
        className,
      )}
      {...rest}
    >
      {children}
    </span>
  );
}
