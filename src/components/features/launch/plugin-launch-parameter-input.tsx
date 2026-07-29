import { SettingsInput } from "#/components/features/settings/settings-input";

export interface PluginLaunchParameterInputProps {
  pluginIndex: number;
  paramKey: string;
  paramValue: unknown;
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
  onParameterChange,
}: PluginLaunchParameterInputProps) {
  const inputId = `plugin-${pluginIndex}-param-${paramKey}`;

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
        <span className="text-sm">{paramKey}</span>
      </label>
    );
  }

  if (typeof paramValue === "number") {
    return (
      <SettingsInput
        testId={inputId}
        name={`plugin-${pluginIndex}-param-${paramKey}`}
        type="number"
        label={paramKey}
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

  return (
    <SettingsInput
      testId={inputId}
      name={`plugin-${pluginIndex}-param-${paramKey}`}
      type="text"
      label={paramKey}
      value={String(paramValue ?? "")}
      className="w-full"
      onChange={(value) => onParameterChange(pluginIndex, paramKey, value)}
    />
  );
}
