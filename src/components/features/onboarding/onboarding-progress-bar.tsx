import { cn } from "#/utils/utils";

interface OnboardingProgressBarProps {
  /** Index of the current step (0-based). */
  currentStep: number;
  /** Total number of steps in the flow. */
  totalSteps: number;
  className?: string;
}

/**
 * Segmented progress bar rendered at the top of the onboarding modal.
 * Each step is its own pill that fills in as the user moves forward,
 * giving an at-a-glance sense of how far they have to go.
 */
export function OnboardingProgressBar({
  currentStep,
  totalSteps,
  className,
}: OnboardingProgressBarProps) {
  return (
    <div
      role="progressbar"
      aria-valuemin={1}
      aria-valuemax={totalSteps}
      aria-valuenow={currentStep + 1}
      data-testid="onboarding-progress-bar"
      className={cn("flex w-full items-center gap-2", className)}
    >
      {Array.from({ length: totalSteps }, (_, index) => {
        const isCompleted = index < currentStep;
        const isCurrent = index === currentStep;
        return (
          <div
            key={index}
            data-testid={`onboarding-progress-step-${index}`}
            data-state={
              isCompleted ? "completed" : isCurrent ? "current" : "upcoming"
            }
            className={cn(
              "h-1.5 flex-1 rounded-full transition-colors duration-300",
              isCompleted || isCurrent ? "bg-white" : "bg-white/15",
            )}
          />
        );
      })}
    </div>
  );
}
