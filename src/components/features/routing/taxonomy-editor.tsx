import React from "react";
import { useTranslation } from "react-i18next";
import type {
  RoutingLabel,
  RoutingTaxonomy,
} from "#/api/routing-service/routing-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { I18nKey } from "#/i18n/declaration";

export function buildClassifierPrompt(
  workTypes: RoutingLabel[],
  sensitivities: RoutingLabel[],
): string {
  const workLines = workTypes
    .map((item) => `- ${item.id}: ${item.name} — ${item.description}`)
    .join("\n");
  const sensLines = sensitivities
    .map((item) => `- ${item.id}: ${item.name} — ${item.description}`)
    .join("\n");
  return `Classify the task into exactly one work_type id, one sensitivity id, and a complexity of low, medium, or high.\nWork types:\n${workLines}\n\nSensitivities:\n${sensLines}\n`;
}

export interface TaxonomyEditorProps {
  taxonomy: RoutingTaxonomy;
  onChange: (
    next: Pick<RoutingTaxonomy, "work_types" | "sensitivities">,
  ) => void;
  onReset: () => void;
}

function LabelList({
  title,
  items,
  testId,
  addLabel,
  onChange,
  onAdd,
}: {
  title: string;
  items: RoutingLabel[];
  testId: string;
  addLabel: string;
  onChange: (items: RoutingLabel[]) => void;
  onAdd: () => void;
}) {
  const { t } = useTranslation("openhands");
  return (
    <div data-testid={testId} className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h4 className="text-sm text-white">{title}</h4>
        <BrandButton
          type="button"
          variant="tertiary"
          testId={`${testId}-add`}
          onClick={onAdd}
        >
          {addLabel}
        </BrandButton>
      </div>
      {items.map((item, index) => (
        <div
          key={item.id}
          className="grid gap-2 rounded-lg border border-[var(--oh-border)] p-2"
        >
          <SettingsInput
            testId={`${testId}-name-${item.id}`}
            name={`${testId}-name-${item.id}`}
            label={t(I18nKey.ROUTING$NAME)}
            type="text"
            value={item.name}
            onChange={(value: string) => {
              const next = items.map((row, rowIndex) =>
                rowIndex === index ? { ...row, name: value } : row,
              );
              onChange(next);
            }}
          />
          <SettingsInput
            testId={`${testId}-description-${item.id}`}
            name={`${testId}-description-${item.id}`}
            label={t(I18nKey.ROUTING$DESCRIPTION)}
            type="text"
            value={item.description}
            onChange={(value: string) => {
              const next = items.map((row, rowIndex) =>
                rowIndex === index ? { ...row, description: value } : row,
              );
              onChange(next);
            }}
          />
          <BrandButton
            type="button"
            variant="ghost-danger"
            testId={`${testId}-delete-${item.id}`}
            onClick={() =>
              onChange(items.filter((_, rowIndex) => rowIndex !== index))
            }
          >
            {t(I18nKey.ROUTING$DELETE)}
          </BrandButton>
        </div>
      ))}
    </div>
  );
}

export function TaxonomyEditor({
  taxonomy,
  onChange,
  onReset,
}: TaxonomyEditorProps) {
  const { t } = useTranslation("openhands");
  const prompt = buildClassifierPrompt(
    taxonomy.work_types,
    taxonomy.sensitivities,
  );
  return (
    <section
      data-testid="routing-taxonomy-editor"
      className="flex flex-col gap-4"
    >
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-white">
          {t(I18nKey.ROUTING$TAXONOMY)}
        </h3>
        <BrandButton
          type="button"
          variant="tertiary"
          testId="routing-taxonomy-reset"
          onClick={onReset}
        >
          {t(I18nKey.ROUTING$RESET_DEFAULTS)}
        </BrandButton>
      </div>
      <LabelList
        title={t(I18nKey.ROUTING$WORK_TYPE)}
        items={taxonomy.work_types}
        testId="routing-work-types"
        addLabel={t(I18nKey.ROUTING$ADD_WORK_TYPE)}
        onAdd={() =>
          onChange({
            work_types: [
              ...taxonomy.work_types,
              {
                id: `work-${taxonomy.work_types.length + 1}`,
                name: "",
                description: "",
              },
            ],
            sensitivities: taxonomy.sensitivities,
          })
        }
        onChange={(work_types) =>
          onChange({ work_types, sensitivities: taxonomy.sensitivities })
        }
      />
      <LabelList
        title={t(I18nKey.ROUTING$SENSITIVITY)}
        items={taxonomy.sensitivities}
        testId="routing-sensitivities"
        addLabel={t(I18nKey.ROUTING$ADD_SENSITIVITY)}
        onAdd={() =>
          onChange({
            work_types: taxonomy.work_types,
            sensitivities: [
              ...taxonomy.sensitivities,
              {
                id: `sensitivity-${taxonomy.sensitivities.length + 1}`,
                name: "",
                description: "",
              },
            ],
          })
        }
        onChange={(sensitivities) =>
          onChange({ work_types: taxonomy.work_types, sensitivities })
        }
      />
      <div>
        <h4 className="mb-2 text-sm text-white">
          {t(I18nKey.ROUTING$PROMPT_PREVIEW)}
        </h4>
        <pre
          data-testid="routing-classifier-prompt"
          className="whitespace-pre-wrap rounded-lg bg-surface-raised p-3 text-xs text-[var(--oh-muted)]"
        >
          {prompt}
        </pre>
      </div>
    </section>
  );
}
