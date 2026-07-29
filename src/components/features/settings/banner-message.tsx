import { cn } from "#/utils/utils";

interface BannerMessageProps {
  message: string;
  className?: string;
}

export function BannerMessage({ message, className }: BannerMessageProps) {
  return (
    <p className={cn("text-xs font-medium text-black", className)}>{message}</p>
  );
}
