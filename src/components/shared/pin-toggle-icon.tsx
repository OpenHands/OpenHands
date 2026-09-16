import PillIcon from "#/icons/pill.svg?react";
import PillFillIcon from "#/icons/pill-fill.svg?react";
import { dropdownMenuPinIconWrapperClassName } from "#/utils/dropdown-classes";
import { cn } from "#/utils/utils";

interface PinToggleIconProps {
  pinned: boolean;
  className?: string;
}

/** Pin/unpin glyph pair sized to match visually in dropdown rows. */
export function PinToggleIcon({ pinned, className }: PinToggleIconProps) {
  return (
    <span
      className={cn("ml-auto", dropdownMenuPinIconWrapperClassName, className)}
      aria-hidden
    >
      {pinned ? (
        <PillFillIcon className="h-4.5 w-4.5" />
      ) : (
        <PillIcon className="h-4.5 w-4.5" />
      )}
    </span>
  );
}
