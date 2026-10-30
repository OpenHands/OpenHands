import { SettingsInput } from "#/components/features/settings/settings-input";
import { cn } from "#/utils/utils";

export interface PluginLaunchParameterInputProps {
  pluginIndex: number;
  paramKey: string;
  paramValue: unknown;
  /** Manifest schema label (title); falls back to the key when absent. */
  label?: string;
  /** Manifest schema options: renders a select instead of free text. */
  options?: string[];
  /** Manifest schema description, shown as helper text. */
  helpText?: string;
  /** UI hint: "textarea" renders a multi-line input for string fields. */
  widget?: "textarea";
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
  helpText,
  widget,
  onParameterChange,
}: PluginLaunchParameterInputProps) {
  const inputId = `plugin-${pluginIndex}-param-${paramKey}`;
  const displayLabel = label ?? paramKey;

  if (options && options.length > 0) {
    return (
      <label htmlFor={inputId} className="flex flex-col gap-1.5 text-sm">
        <span className="font-medium text-contrast">{displayLabel}</span>
        <select
          id={inputId}
          data-testid={inputId}
          value={String(paramValue ?? "")}
          onChange={(e) =>
            onParameterChange(pluginIndex, paramKey, e.target.value)
          }
          className="w-full rounded-md border border-tertiary-light bg-input px-3 py-2 text-contrast focus:border-primary focus:outline-none"
        >
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

  if (typeof paramValue === "boolean") {
    return (
      <label
        htmlFor={inputId}
        className="flex w-full cursor-pointer items-center gap-2.5"
      >
        <input
          id={inputId}
          data-testid={inputId}
          type="checkbox"
          checked={paramValue}
          onChange={(e) =>
            onParameterChange(pluginIndex, paramKey, e.target.checked)
          }
          className="h-4 w-4 shrink-0 rounded"
        />
        <span className="text-sm">{displayLabel}</span>
      </label>
    );
  }

  if (typeof paramValue === "number") {
    return (
      <SettingsInput
        testId={inputId}
        name={`plugin-${pluginIndex}-param-${paramKey}`}
        type="number"
        label={displayLabel}
        value={String(paramValue)}
        className="w-full"
        onChange={(value) =>
          onParameterChange(
            pluginIndex,
            paramKey,
            value === "" ? 0 : parseFloat(value) || 0,
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
            "w-full rounded-md border border-tertiary-light bg-input px-3 py-2",
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
