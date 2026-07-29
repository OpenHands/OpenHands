import LinkExternalIcon from "#/icons/link-external.svg?react";
import { cn } from "#/utils/utils";

interface GitExternalLinkIconProps {
  className?: string;
}

export function GitExternalLinkIcon({ className }: GitExternalLinkIconProps) {
  return (
    <div
      className={cn(
        "w-3 h-3 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300 absolute right-0 top-1/2 -translate-y-1/2 h-full w-10.5 pr-2.5 justify-end git-external-link-icon",
        className,
      )}
    >
      <LinkExternalIcon width={12} height={12} color="white" />
    </div>
  );
}
