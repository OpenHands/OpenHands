import { StepOption } from "./step-option";
import { StepInput } from "./step-input";

export interface Option {
  id: string;
  label: string;
}

export interface InputField {
  id: string;
  label: string;
}

interface StepContentProps {
  options?: Option[];
  inputFields?: InputField[];
  selectedOptionIds: string[];
  inputValues?: Record<string, string>;
  onSelectOption: (optionId: string) => void;
  onInputChange?: (fieldId: string, value: string) => void;
}

export function StepContent({
  options,
  inputFields,
  selectedOptionIds,
  inputValues = {},
  onSelectOption,
  onInputChange,
}: StepContentProps) {
  return (
    <div
      data-testid="step-content"
      className="flex flex-col mt-8 mb-8 gap-[12px] w-full"
    >
      {options?.map((option) => (
        <StepOption
          key={option.id}
          id={option.id}
          label={option.label}
          selected={selectedOptionIds.includes(option.id)}
          onClick={() => onSelectOption(option.id)}
        />
      ))}
      {inputFields?.map((field) => (
        <StepInput
          key={field.id}
          id={field.id}
          label={field.label}
          value={inputValues[field.id] || ""}
          onChange={(value) => onInputChange?.(field.id, value)}
        />
      ))}
    </div>
  );
}
