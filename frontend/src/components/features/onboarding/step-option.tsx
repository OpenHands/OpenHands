import { cn } from "#/utils/utils";
import { Typography } from "#/ui/typography";

interface StepOptionProps {
  id: string;
  label: string;
  selected: boolean;
  onClick: () => void;
}

export function StepOption({ id, label, selected, onClick }: StepOptionProps) {
  return (
    <button
      data-testid={`step-option-${id}`}
      type="button"
      tabIndex={0}
      onClick={onClick}
      className={cn(
        "min-h-10 w-full rounded-md border text-left px-4 py-2.5 transition-colors text-white cursor-pointer",
        selected
          ? "border-white bg-[#3a3a3a]"
          : "border-[#3a3a3a] hover:bg-[#3a3a3a]",
      )}
    >
      <Typography.Text className="text-sm font-medium text-content">
        {label}
      </Typography.Text>
    </button>
  );
}
