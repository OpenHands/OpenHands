import { Typography } from "#/ui/typography";
import { cn } from "#/utils/utils";

interface StepHeaderProps {
  title: string;
  currentStep: number;
  totalSteps: number;
}

function StepHeader({ title, currentStep, totalSteps }: StepHeaderProps) {
  return (
    <div data-testid="step-header" className="flex flex-col items-center gap-2">
      <div className="flex justify-center gap-2 mb-2">
        {Array.from({ length: totalSteps }).map((_, index) => (
          <div
            key={index}
            className={cn(
              "w-[6px] h-[4px] rounded-full transition-colors",
              index < currentStep ? "bg-white" : "bg-neutral-600",
            )}
          />
        ))}
      </div>
      <Typography.Text className="text-2xl font-semibold text-content text-center">
        {title}
      </Typography.Text>
    </div>
  );
}

export default StepHeader;
