import { useTranslation } from "react-i18next";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";
import { I18nKey } from "#/i18n/declaration";

export interface ReportViewProps {
  markdown: string;
}

export function ReportView({ markdown }: ReportViewProps) {
  const { t } = useTranslation("openhands");

  return (
    <section data-testid="feature-dev-report" className="flex flex-col gap-2">
      <h3 className="text-sm font-medium text-white">
        {t(I18nKey.FEATURE_DEV$REPORT)}
      </h3>
      <div className="rounded-xl bg-base-secondary p-3 text-sm text-white">
        <MarkdownRenderer>{markdown}</MarkdownRenderer>
      </div>
    </section>
  );
}
