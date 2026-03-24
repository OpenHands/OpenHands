import { isValidEmail } from "#/utils/input-validation";
import { cn } from "#/utils/utils";

// Email validation pattern - must match EMAIL_REGEX in input-validation.ts
const EMAIL_PATTERN = "[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}";

interface FormInputProps {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  type?: "text" | "email";
  rows?: number;
  required?: boolean;
  showError?: boolean;
}

export function FormInput({
  id,
  label,
  value,
  onChange,
  placeholder,
  type = "text",
  rows,
  required = false,
  showError = false,
}: FormInputProps) {
  const inputId = `form-input-${id}`;
  const isEmailInvalid =
    type === "email" && !!value.trim() && !isValidEmail(value.trim());
  const hasError = showError && ((required && !value.trim()) || isEmailInvalid);
  const baseClassName = cn(
    "w-full min-h-10 rounded border border-[#242424] px-3 py-2 text-sm leading-5 text-white placeholder:text-[#8C8C8C] placeholder:leading-5 focus:outline-none transition-colors focus:border-white",
  );

  return (
    <div className="flex flex-col gap-1.5 w-full">
      <label
        htmlFor={inputId}
        className="text-sm font-medium leading-5 text-[#FAFAFA] cursor-pointer"
      >
        {label}
      </label>
      {rows ? (
        <textarea
          id={inputId}
          data-testid={inputId}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          rows={rows}
          required={required}
          aria-required={required}
          aria-invalid={hasError}
          aria-label={label}
          className={cn(baseClassName, "h-auto resize-none bg-transparent")}
        />
      ) : (
        <input
          id={inputId}
          data-testid={inputId}
          type={type}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          pattern={type === "email" ? EMAIL_PATTERN : undefined}
          required={required}
          aria-required={required}
          aria-invalid={hasError}
          aria-label={label}
          className={cn(baseClassName, "bg-[#1F1F1F66]")}
        />
      )}
    </div>
  );
}
