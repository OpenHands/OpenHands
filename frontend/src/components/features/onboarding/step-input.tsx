interface StepInputProps {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
}

export function StepInput({ id, label, value, onChange }: StepInputProps) {
  return (
    <div className="flex flex-col gap-1.5 w-full">
      <label
        htmlFor={`step-input-${id}`}
        className="text-sm font-medium text-neutral-400 cursor-pointer"
      >
        {label}
      </label>
      <input
        id={`step-input-${id}`}
        data-testid={`step-input-${id}`}
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-md border border-[#3a3a3a] bg-transparent px-4 py-2.5 text-sm text-white placeholder:text-neutral-500 focus:border-white focus:outline-none transition-colors"
      />
    </div>
  );
}
