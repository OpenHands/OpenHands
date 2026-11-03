import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import DocumentIcon from "#/icons/document.svg?react";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";
import { SectionCard } from "./section-card";

interface DescriptionSectionProps {
  description: string;
}

/** The automation's description, rendered as Markdown (with Mermaid diagrams). */
export function DescriptionSection({ description }: DescriptionSectionProps) {
  const { t } = useTranslation("openhands");

  return (
    <SectionCard
      icon={<DocumentIcon className="size-4" />}
      title={t(I18nKey.AUTOMATIONS$DETAIL$DESCRIPTION)}
    >
      <div
        className="text-sm leading-6 text-content"
        data-testid="automation-description-content"
      >
        <MarkdownRenderer content={description} includeStandard />
      </div>
    </SectionCard>
  );
}
