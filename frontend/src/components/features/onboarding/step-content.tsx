import { StepOption } from "./step-option";

export interface Option {
  id: string;
  label: string;
}

interface StepContentProps {
  options: Option[];
  selectedOptionId: string | null;
  onSelectOption: (optionId: string) => void;
}

export function StepContent({
  options,
  selectedOptionId,
  onSelectOption,
}: StepContentProps) {
  return (
    <div
      data-testid="step-content"
      className="flex flex-col mt-8 mb-8 gap-[12px] w-full"
    >
      {options.map((option) => (
        <StepOption
          key={option.id}
          id={option.id}
          label={option.label}
          selected={selectedOptionId === option.id}
          onClick={() => onSelectOption(option.id)}
        />
      ))}
    </div>
  );
}
