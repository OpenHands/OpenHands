import { useTranslation } from "react-i18next";
import type { StandardsRunResult } from "#/api/standards-service/standards-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";

export interface StandardsConsoleProps {
  root: string;
  onRootChange: (value: string) => void;
  onRun: () => void;
  isBusy?: boolean;
  result: StandardsRunResult | null;
}

export function StandardsConsole({
  root,
  onRootChange,
  onRun,
  isBusy = false,
  result,
}: StandardsConsoleProps) {
  const { t } = useTranslation("openhands");

  return (
    <section
      data-testid="standards-console"
      className="flex flex-col gap-3 rounded-xl bg-base-secondary p-4"
    >
      <Typography variant="h3">{t(I18nKey.STANDARDS$CONSOLE)}</Typography>
      <SettingsInput
        testId="standards-root-input"
        name="standards-root"
        label={t(I18nKey.STANDARDS$ROOT)}
        type="text"
        value={root}
        onChange={onRootChange}
      />
      <BrandButton
        type="button"
        variant="primary"
        testId="standards-run"
        isDisabled={isBusy || !root.trim()}
        onClick={onRun}
      >
        {t(I18nKey.STANDARDS$RUN)}
      </BrandButton>
      {result ? (
        <>
          <div className="grid grid-cols-2 gap-2 text-sm text-white sm:grid-cols-4">
            <p data-testid="standards-run-id">
              {t(I18nKey.STANDARDS$RUN_ID)}: {result.run_id}
            </p>
            <p data-testid="standards-files-scanned">
              {t(I18nKey.STANDARDS$FILES_SCANNED)}:{" "}
              {result.summary.files_scanned}
            </p>
            <p data-testid="standards-duration">
              {t(I18nKey.STANDARDS$DURATION)}: {result.duration_ms}
            </p>
            <p data-testid="standards-summary-warning">
              {t(I18nKey.STANDARDS$SEVERITY_WARNING)}: {result.summary.warning}
            </p>
            <p data-testid="standards-summary-error">
              {t(I18nKey.STANDARDS$SEVERITY_ERROR)}: {result.summary.error}
            </p>
            <p data-testid="standards-summary-info">
              {t(I18nKey.STANDARDS$SEVERITY_INFO)}: {result.summary.info}
            </p>
          </div>
          {result.violations.length === 0 ? (
            <p
              data-testid="standards-violations-empty"
              className="text-sm text-tertiary-light"
            >
              {t(I18nKey.STANDARDS$NO_VIOLATIONS)}
            </p>
          ) : (
            <table
              data-testid="standards-violations-table"
              className="w-full text-left text-xs text-white"
            >
              <thead>
                <tr className="text-tertiary-light">
                  <th>{t(I18nKey.STANDARDS$RULE_ID)}</th>
                  <th>{t(I18nKey.STANDARDS$FILE)}</th>
                  <th>{t(I18nKey.STANDARDS$MESSAGE)}</th>
                  <th>{t(I18nKey.STANDARDS$REMEDIATION)}</th>
                  <th>{t(I18nKey.STANDARDS$ACTION)}</th>
                </tr>
              </thead>
              <tbody>
                {result.violations.map((item, index) => (
                  <tr
                    key={`${item.rule_id}-${item.file}-${item.line}-${index}`}
                    data-testid={`standards-violation-${item.rule_id}`}
                  >
                    <td>{item.rule_id}</td>
                    <td>
                      {item.file}
                      {item.line != null ? `:${item.line}` : ""}
                    </td>
                    <td>{item.message}</td>
                    <td>{item.remediation}</td>
                    <td>{item.action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      ) : null}
    </section>
  );
}
