import { forwardRef } from "react";
import { cn } from "#/utils/utils";
import { formControlSettingsFieldClassName } from "#/utils/form-control-classes";
import { OptionalTag } from "./optional-tag";

interface SettingsInputProps {
  testId?: string;
  name?: string;
  label: string;
  type: React.HTMLInputTypeAttribute;
  defaultValue?: string;
  value?: string;
  placeholder?: string;
  showOptionalTag?: boolean;
  isDisabled?: boolean;
  startContent?: React.ReactNode;
  className?: string;
  onChange?: (value: string) => void;
  onKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  required?: boolean;
  min?: number;
  max?: number;
  step?: number;
  pattern?: string;
  /** Validation message shown when pattern doesn't match */
  title?: string;
  labelClassName?: string;
  /** ARIA describedby attribute for accessibility */
  ariaDescribedBy?: string;
  /** ARIA invalid attribute for accessibility */
  ariaInvalid?: boolean;
  /**
   * Validation error message. When set, the input gets a red border and
   * the message is rendered below it. Also sets aria-invalid automatically.
   */
  error?: string;
  /** Renders a red asterisk next to the label to mark the field as required. */
  showRequiredTag?: boolean;
  onBlur?: () => void;
  /** Extra classes merged onto the `<input>` element. */
  inputClassName?: string;
}

export const SettingsInput = forwardRef<HTMLInputElement, SettingsInputProps>(
  function SettingsInput(
    {
      testId,
      name,
      label,
      type,
      defaultValue,
      value,
      placeholder,
      showOptionalTag,
      isDisabled,
      startContent,
      className,
      onChange,
      onKeyDown,
      required,
      min,
      max,
      step,
      pattern,
      title,
      labelClassName,
      ariaDescribedBy,
      ariaInvalid,
      error,
      showRequiredTag,
      onBlur,
      inputClassName,
    },
    ref,
  ) {
    const errorId = error && testId ? `${testId}-error` : undefined;
    return (
      <label className={cn("flex flex-col gap-2.5 w-full min-w-0", className)}>
        <div className="flex items-center gap-2">
          {startContent}
          <span className={cn("text-sm", labelClassName)}>{label}</span>
          {showRequiredTag && (
            <span className="text-red-400 text-sm leading-none" aria-hidden>
              *
            </span>
          )}
          {showOptionalTag && <OptionalTag />}
        </div>
        <input
          ref={ref}
          data-testid={testId}
          onChange={(e) => onChange?.(e.target.value)}
          onKeyDown={onKeyDown}
          onBlur={onBlur}
          name={name}
          disabled={isDisabled}
          type={type}
          defaultValue={defaultValue}
          value={value}
          placeholder={placeholder}
          min={min}
          max={max}
          step={step}
          required={required}
          pattern={pattern}
          title={title}
          aria-describedby={errorId ?? ariaDescribedBy}
          aria-invalid={!!error || ariaInvalid}
          className={cn(
            formControlSettingsFieldClassName,
            "disabled:bg-[var(--oh-surface-raised)] disabled:border-[var(--oh-border-subtle)]",
            error && "border-red-500",
            inputClassName,
          )}
        />
        {error && (
          <p
            id={errorId}
            role="alert"
            data-testid={testId ? `${testId}-error` : undefined}
            className="text-xs text-red-400 -mt-1"
          >
            {error}
          </p>
        )}
      </label>
    );
  },
);
