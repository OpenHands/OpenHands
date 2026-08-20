import { ChevronDown, ChevronRight } from "lucide-react";
import { Typography } from "#/ui/typography";
import { cn } from "#/utils/utils";

interface ToggleButtonProps {
  title: string;
  isExpanded: boolean;
  onClick: () => void;
  className?: string;
}

export function ToggleButton({
  title,
  isExpanded,
  onClick,
  className,
}: ToggleButtonProps) {
  return (
    <button
      type="button"
      data-testid="toggle-button"
      onClick={onClick}
      className={cn(
        "w-full py-3 px-3 text-left flex items-center justify-between hover:bg-tertiary transition-colors",
        className,
      )}
    >
      <div className="flex items-center">
        <Typography.Text className="font-bold text-content-2">
          {title}
        </Typography.Text>
      </div>
      <Typography.Text className="text-[var(--oh-text-tertiary)]">
        {isExpanded ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
      </Typography.Text>
    </button>
  );
}
