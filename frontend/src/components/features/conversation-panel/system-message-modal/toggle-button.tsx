import { ChevronDown, ChevronRight } from "lucide-react";
import { Typography } from "#/ui/typography";

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
      onClick={onClick}
      className={`w-full py-3 px-2 text-left flex items-center justify-between hover:bg-surface-hover transition-colors ${className || ""}`}
    >
      <div className="flex items-center">
        <Typography.Text className="font-bold text-content">
          {title}
        </Typography.Text>
      </div>
      <Typography.Text className="text-muted">
        {isExpanded ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
      </Typography.Text>
    </button>
  );
}
