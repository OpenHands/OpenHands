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
      data-testid="toggle-button"
      onClick={onClick}
      className={`w-full py-3 px-2 text-left flex items-center justify-between hover:bg-gray-700 transition-colors ${className || ""}`}
    >
      <div className="flex items-center">
        <Typography.Text className="font-bold text-gray-100">
          {title}
        </Typography.Text>
      </div>
      <Typography.Text className="text-gray-300">
        {isExpanded ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
      </Typography.Text>
    </button>
  );
}
