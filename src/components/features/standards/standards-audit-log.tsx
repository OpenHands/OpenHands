import { useTranslation } from "react-i18next";
import type { StandardsAuditPage } from "#/api/standards-service/standards-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { formControlFieldClassName } from "#/utils/form-control-classes";

export interface StandardsAuditLogProps {
  page: StandardsAuditPage | undefined;
  plugin: string;
  severity: string;
  file: string;
  onPluginChange: (value: string) => void;
  onSeverityChange: (value: string) => void;
  onFileChange: (value: string) => void;
  onLoadOlder: () => void;
}

export function StandardsAuditLog({
  page,
  plugin,
  severity,
  file,
  onPluginChange,
  onSeverityChange,
  onFileChange,
  onLoadOlder,
}: StandardsAuditLogProps) {
  const { t } = useTranslation("openhands");
  const items = page?.items ?? [];

  return (
    <section
      data-testid="standards-audit-log"
      className="flex flex-col gap-3 rounded-xl bg-base-secondary p-4"
    >
      <Typography variant="h3">{t(I18nKey.STANDARDS$AUDIT)}</Typography>
      <div className="flex flex-wrap gap-2">
        <SettingsInput
          testId="standards-audit-plugin"
          name="standards-audit-plugin"
          label={t(I18nKey.STANDARDS$FILTER_PLUGIN)}
          type="text"
          value={plugin}
          onChange={onPluginChange}
        />
        <label className="flex flex-col gap-1 text-xs text-tertiary-light">
          {t(I18nKey.STANDARDS$FILTER_SEVERITY)}
          <select
            data-testid="standards-audit-severity"
            className={formControlFieldClassName}
            value={severity}
            onChange={(event) => onSeverityChange(event.target.value)}
          >
            <option value="">{t(I18nKey.STANDARDS$ALL)}</option>
            <option value="info">{t(I18nKey.STANDARDS$SEVERITY_INFO)}</option>
            <option value="warning">
              {t(I18nKey.STANDARDS$SEVERITY_WARNING)}
            </option>
            <option value="error">{t(I18nKey.STANDARDS$SEVERITY_ERROR)}</option>
          </select>
        </label>
        <SettingsInput
          testId="standards-audit-file"
          name="standards-audit-file"
          label={t(I18nKey.STANDARDS$FILTER_FILE)}
          type="text"
          value={file}
          onChange={onFileChange}
        />
      </div>
      {items.length === 0 ? (
        <p
          data-testid="standards-audit-empty"
          className="text-sm text-tertiary-light"
        >
          {t(I18nKey.STANDARDS$NO_AUDIT)}
        </p>
      ) : (
        <table className="w-full text-left text-xs text-white">
          <thead>
            <tr className="text-tertiary-light">
              <th>{t(I18nKey.STANDARDS$RUN_ID)}</th>
              <th>{t(I18nKey.STANDARDS$FILTER_PLUGIN)}</th>
              <th>{t(I18nKey.STANDARDS$SEVERITY)}</th>
              <th>{t(I18nKey.STANDARDS$FILE)}</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.id} data-testid={`standards-audit-row-${item.id}`}>
                <td>{item.run_id}</td>
                <td>{item.plugin_name}</td>
                <td>{item.severity}</td>
                <td>
                  {item.file}
                  {item.line != null ? `:${item.line}` : ""}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {page?.next_before_id ? (
        <BrandButton
          type="button"
          variant="secondary"
          testId="standards-audit-load-older"
          onClick={onLoadOlder}
        >
          {t(I18nKey.STANDARDS$LOAD_OLDER)}
        </BrandButton>
      ) : null}
    </section>
  );
}
