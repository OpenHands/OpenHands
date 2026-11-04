import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import DocumentIcon from "#/icons/document.svg?react";
import { MarkdownRenderer } from "#/components/features/markdown/markdown-renderer";
import { MermaidDiagram } from "#/components/features/markdown/mermaid-diagram";
import { SectionCard } from "./section-card";

interface DescriptionSectionProps {
  description: string;
  /** Mermaid source drawn under the description, e.g. a template's diagram. */
  diagram?: string | null;
}

/** The automation's description, rendered as Markdown (with Mermaid diagrams). */
export function DescriptionSection({
  description,
  diagram,
}: DescriptionSectionProps) {
  const { t } = useTranslation("openhands");

  return (
    <SectionCard
      icon={<DocumentIcon className="size-4" />}
      title={t(I18nKey.AUTOMATIONS$DETAIL$DESCRIPTION)}
    >
      <div
        className="flex flex-col gap-4 text-sm leading-6 text-content"
        data-testid="automation-description-content"
      >
        <MarkdownRenderer content={description} includeStandard />
        {diagram ? <MermaidDiagram source={diagram} fallback={null} /> : null}
      </div>
    </SectionCard>
  );
}
