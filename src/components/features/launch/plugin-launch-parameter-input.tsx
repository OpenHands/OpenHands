import { SettingsInput } from "#/components/features/settings/settings-input";
import { cn } from "#/utils/utils";

const SELECT_PLACEHOLDER = "— select —";
// Suffix shown on the disabled option that surfaces a badge value not in the
// schema's enum, so display and submission agree.
const STALE_VALUE_SUFFIX = "(current)";
export interface PluginLaunchParameterInputProps {
  pluginIndex: number;
  paramKey: string;
  paramValue: unknown;
  /** Manifest schema label (title); falls back to the key when absent. */
  label?: string;
  /** Manifest schema options: renders a select instead of free text. */
  options?: string[];
  /** Raw (typed) enum values paired with `options`, to avoid string coercion. */
  enumValues?: unknown[];
  /** Manifest schema description, shown as helper text. */
  helpText?: string;
  /** UI hint: "textarea" renders a multi-line input for string fields. */
  widget?: "textarea";
  /**
   * Schema-declared type, used to pick the widget when the runtime value
   * doesn't reflect it (e.g. an integer rendered as "5"). When absent, the
   * widget is inferred from `typeof paramValue`.
   */
  fieldType?: "string" | "number" | "integer" | "boolean";
  /** Whether the field is required (renders the asterisk + placeholder). */
  required?: boolean;
  onParameterChange: (
    pluginIndex: number,
    paramKey: string,
    value: unknown,
  ) => void;
}

export function PluginLaunchParameterInput({
  pluginIndex,
  paramKey,
  paramValue,
  label,
  options,
  enumValues,
  helpText,
  widget,
  fieldType,
  required = false,
  onParameterChange,
}: PluginLaunchParameterInputProps) {
  const inputId = `plugin-${pluginIndex}-param-${paramKey}`;
  const displayLabel = required
    ? `${label ?? paramKey} *`
    : (label ?? paramKey);
  // Schema type wins over runtime typeof: an integer rendered as "5" is still
  // an integer, and a boolean with no value is still a checkbox.
  const effectiveType =
    fieldType ??
    (typeof paramValue === "number"
      ? "number"
      : typeof paramValue === "boolean"
        ? "boolean"
        : "string");

  if (options && options.length > 0) {
    // Recover the raw enum value (number/boolean/string) by matching the
    // selected display string; fall back to the string for unknown values.
    const resolveEnumValue = (selected: string): unknown => {
      if (selected === "") return undefined;
      const index = enumValues
        ? enumValues.findIndex((v) => String(v) === selected)
        : -1;
      return index >= 0 ? enumValues![index] : selected;
    };
    const currentValueStr = String(paramValue ?? "");
    const valueMatchesOption = options.includes(currentValueStr);
    // When the badge pre-encodes a value not in the schema's enum, render a
    // disabled placeholder carrying the raw value so display and submission
    // agree — otherwise the select shows "— select —" while pluginConfigs
    // still holds the badge value.
    const showStaleValueOption = currentValueStr !== "" && !valueMatchesOption;
    return (
      <label htmlFor={inputId} className="flex flex-col gap-1.5 text-sm">
        <span className="font-medium text-contrast">{displayLabel}</span>
        <select
          id={inputId}
          data-testid={inputId}
          value={currentValueStr}
          onChange={(e) =>
            onParameterChange(
              pluginIndex,
              paramKey,
              resolveEnumValue(e.target.value),
            )
          }
          className="w-full rounded-md border border-tertiary-light bg-tertiary px-3 py-2 text-contrast focus:border-primary focus:outline-none"
        >
          {showStaleValueOption && (
            <option value={currentValueStr} disabled>
              {currentValueStr} {STALE_VALUE_SUFFIX}
            </option>
          )}
          {!required && !showStaleValueOption && (
            <option value={SELECT_PLACEHOLDER}>{SELECT_PLACEHOLDER}</option>
          )}
          {options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
        {helpText && (
          <span className="text-xs text-tertiary leading-relaxed">
            {helpText}
          </span>
        )}
      </label>
    );
  }

  if (effectiveType === "boolean") {
    return (
      <label
        htmlFor={inputId}
        className="flex w-full cursor-pointer items-center gap-2.5"
      >
        <input
          id={inputId}
          data-testid={inputId}
          type="checkbox"
          checked={Boolean(paramValue)}
          onChange={(e) =>
            onParameterChange(pluginIndex, paramKey, e.target.checked)
          }
          className="h-4 w-4 shrink-0 rounded"
        />
        <span className="text-sm">{displayLabel}</span>
      </label>
    );
  }

  if (effectiveType === "integer" || effectiveType === "number") {
    const isInteger = effectiveType === "integer";
    return (
      <SettingsInput
        testId={inputId}
        name={`plugin-${pluginIndex}-param-${paramKey}`}
        type="number"
        label={displayLabel}
        value={
          paramValue === undefined || paramValue === null
            ? ""
            : String(paramValue)
        }
        className="w-full"
        step={isInteger ? 1 : undefined}
        onChange={(value) =>
          onParameterChange(
            pluginIndex,
            paramKey,
            value === ""
              ? undefined
              : isInteger
                ? parseInt(value, 10) || 0
                : parseFloat(value) || 0,
          )
        }
      />
    );
  }

  if (widget === "textarea") {
    return (
      <label htmlFor={inputId} className="flex flex-col gap-1.5 text-sm">
        <span className="font-medium text-contrast">{displayLabel}</span>
        <textarea
          id={inputId}
          data-testid={inputId}
          value={String(paramValue ?? "")}
          onChange={(e) =>
            onParameterChange(pluginIndex, paramKey, e.target.value)
          }
          rows={4}
          className={cn(
            "w-full rounded-md border border-tertiary-light bg-tertiary px-3 py-2",
            "text-contrast font-mono text-xs",
            "focus:border-primary focus:outline-none",
          )}
        />
        {helpText && (
          <span className="text-xs text-tertiary leading-relaxed">
            {helpText}
          </span>
        )}
      </label>
    );
  }

  return (
    <div className="flex flex-col gap-1">
      <SettingsInput
        testId={inputId}
        name={`plugin-${pluginIndex}-param-${paramKey}`}
        type="text"
        label={displayLabel}
        value={String(paramValue ?? "")}
        className="w-full"
        onChange={(value) => onParameterChange(pluginIndex, paramKey, value)}
      />
      {helpText && <span className="text-xs text-contrast">{helpText}</span>}
    </div>
  );
}
