import React from "react";
import { useTranslation } from "react-i18next";
import { GRAPH_LANGUAGES } from "#/api/graph-service/graph-constants";
import type { GraphConfig } from "#/api/graph-service/graph-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { I18nKey } from "#/i18n/declaration";
import { ToggleSwitch } from "#/ui/toggle-switch";
import { Typography } from "#/ui/typography";

export interface GraphSettingsProps {
  config: GraphConfig;
  onChange: (patch: Partial<GraphConfig>) => void;
  onImport?: (path: string) => void;
}

export function GraphSettings({
  config,
  onChange,
  onImport,
}: GraphSettingsProps) {
  const { t } = useTranslation("openhands");
  const [importPath, setImportPath] = React.useState("");

  return (
    <section
      data-testid="graph-settings"
      className="flex flex-col gap-3 rounded-xl bg-base-secondary p-4"
    >
      <Typography variant="h3">{t(I18nKey.GRAPH$SETTINGS)}</Typography>
      <ToggleSwitch
        enabled={config.enabled}
        label={t(I18nKey.GRAPH$ENABLED)}
        onToggle={() => onChange({ enabled: !config.enabled })}
      />
      <ToggleSwitch
        enabled={config.strict}
        label={t(I18nKey.GRAPH$STRICT)}
        onToggle={() => onChange({ strict: !config.strict })}
      />
      <fieldset data-testid="graph-languages-fieldset">
        <legend className="text-xs text-tertiary-light">
          {t(I18nKey.GRAPH$LANGUAGES)}
        </legend>
        <div className="mt-2 flex flex-wrap gap-3">
          {GRAPH_LANGUAGES.map((language) => {
            const checked = config.languages.includes(language);
            return (
              <label
                key={language}
                className="flex items-center gap-2 text-sm text-white"
              >
                <input
                  type="checkbox"
                  data-testid={`graph-language-${language}`}
                  checked={checked}
                  onChange={() => {
                    const next = checked
                      ? config.languages.filter((item) => item !== language)
                      : [...config.languages, language];
                    onChange({ languages: next });
                  }}
                />
                {language}
              </label>
            );
          })}
        </div>
      </fieldset>
      <SettingsInput
        testId="graph-budget-lines"
        label={t(I18nKey.GRAPH$BUDGET_LINES)}
        type="number"
        value={String(config.graph_budget_lines)}
        onChange={(value) => onChange({ graph_budget_lines: Number(value) })}
      />
      <SettingsInput
        testId="graph-max-files"
        label={t(I18nKey.GRAPH$MAX_FILES)}
        type="number"
        value={String(config.max_context_files)}
        onChange={(value) => onChange({ max_context_files: Number(value) })}
      />
      <SettingsInput
        testId="graph-stale-after"
        label={t(I18nKey.GRAPH$STALE_AFTER)}
        type="number"
        value={String(config.stale_after_minutes)}
        onChange={(value) => onChange({ stale_after_minutes: Number(value) })}
      />
      <p
        data-testid="graph-project-yaml-note"
        className="text-xs text-tertiary-light"
      >
        {t(I18nKey.GRAPH$PROJECT_YAML)}
      </p>
      {onImport ? (
        <div className="flex gap-2">
          <SettingsInput
            testId="graph-import-path"
            label={t(I18nKey.GRAPH$IMPORT)}
            type="text"
            value={importPath}
            onChange={setImportPath}
          />
          <BrandButton
            type="button"
            variant="secondary"
            testId="graph-import-submit"
            onClick={() => importPath && onImport(importPath)}
          >
            {t(I18nKey.GRAPH$IMPORT)}
          </BrandButton>
        </div>
      ) : null}
    </section>
  );
}
