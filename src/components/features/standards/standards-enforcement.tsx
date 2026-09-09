import { useTranslation } from "react-i18next";
import {
  STANDARDS_ACTION_BLOCK,
  STANDARDS_ACTION_WARN,
} from "#/api/standards-service/standards-constants";
import type {
  StandardsAction,
  StandardsConfig,
  StandardsPluginInfo,
} from "#/api/standards-service/standards-types";
import { I18nKey } from "#/i18n/declaration";
import { ToggleSwitch } from "#/ui/toggle-switch";
import { Typography } from "#/ui/typography";
import { formControlFieldClassName } from "#/utils/form-control-classes";

export interface StandardsEnforcementProps {
  config: StandardsConfig;
  plugins: StandardsPluginInfo[];
  onChange: (patch: Partial<StandardsConfig>) => void;
}

export function StandardsEnforcement({
  config,
  plugins,
  onChange,
}: StandardsEnforcementProps) {
  const { t } = useTranslation("openhands");

  const setPluginAction = (name: string, action: StandardsAction) => {
    const next = config.plugins.map((item) =>
      item.name === name ? { ...item, action } : item,
    );
    const known = new Set(next.map((item) => item.name));
    for (const plugin of plugins) {
      if (!known.has(plugin.name)) {
        next.push({ name: plugin.name, enabled: plugin.enabled, action });
      }
    }
    onChange({ plugins: next });
  };

  return (
    <section
      data-testid="standards-enforcement"
      className="flex flex-col gap-3 rounded-xl bg-base-secondary p-4"
    >
      <Typography variant="h3">{t(I18nKey.STANDARDS$ENFORCEMENT)}</Typography>
      <ToggleSwitch
        enabled={config.enforcement.prompt}
        label={t(I18nKey.STANDARDS$PROMPT)}
        onToggle={() =>
          onChange({
            enforcement: {
              ...config.enforcement,
              prompt: !config.enforcement.prompt,
            },
          })
        }
      />
      <ToggleSwitch
        enabled={config.enforcement.automated}
        label={t(I18nKey.STANDARDS$AUTOMATED)}
        onToggle={() =>
          onChange({
            enforcement: {
              ...config.enforcement,
              automated: !config.enforcement.automated,
            },
          })
        }
      />
      <ToggleSwitch
        enabled={config.enforcement.gates}
        label={t(I18nKey.STANDARDS$GATES)}
        onToggle={() =>
          onChange({
            enforcement: {
              ...config.enforcement,
              gates: !config.enforcement.gates,
            },
          })
        }
      />
      {config.project_yaml_source ? (
        <p
          data-testid="standards-project-yaml-notice"
          className="text-xs text-tertiary-light"
        >
          {t(I18nKey.STANDARDS$PROJECT_YAML_NOTICE)}
        </p>
      ) : null}
      <div className="flex flex-col gap-2">
        {plugins.map((plugin) => {
          const entry = config.plugins.find(
            (item) => item.name === plugin.name,
          );
          const action = entry?.action ?? plugin.action;
          return (
            <label
              key={plugin.name}
              className="flex items-center justify-between gap-3 text-sm text-white"
            >
              <span>
                {t(I18nKey.STANDARDS$ACTION)}: {plugin.display_name}
              </span>
              <select
                data-testid={`standards-action-${plugin.name}`}
                className={formControlFieldClassName}
                value={action}
                onChange={(event) =>
                  setPluginAction(
                    plugin.name,
                    event.target.value as StandardsAction,
                  )
                }
              >
                <option value={STANDARDS_ACTION_WARN}>
                  {t(I18nKey.STANDARDS$ACTION_WARN)}
                </option>
                <option value={STANDARDS_ACTION_BLOCK}>
                  {t(I18nKey.STANDARDS$ACTION_BLOCK)}
                </option>
              </select>
            </label>
          );
        })}
      </div>
    </section>
  );
}
