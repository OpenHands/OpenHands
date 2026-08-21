import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import {
  AutomationPreviewField,
  automationPreviewListClassName,
} from "#/components/features/automations/automation-preview-field";
import { previewChipItems } from "#/components/features/automations/automation-preview-chips";
import { setupPreviewFieldIcon } from "#/components/features/automations/automation-preview-icons";
import {
  isPreviewIdentityField,
  sortPreviewFields,
  type PreviewFieldKind,
} from "#/components/features/automations/automation-preview-order";
import {
  collectFields,
  fieldValues,
} from "#/manifests/manifest-local-validation";
import type {
  SetupBlock,
  SetupFieldType,
  SetupFormValues,
} from "#/manifests/types";

export interface SetupReviewStepProps {
  setup: SetupBlock;
  values: SetupFormValues;
  /** Catalog name, shown first when the form has no name/title field. */
  automationName?: string;
}

interface ReviewRow {
  name: string;
  label: string;
  value: string;
  kind: PreviewFieldKind;
  parts: string[];
}

/**
 * Stage 7 — the plain-language summary the user confirms.
 *
 * Rows follow the shared preview order (identity, when, where, inputs, body,
 * add-ons). A manifest declares no order of its own.
 */
export function SetupReviewStep({
  setup,
  values,
  automationName,
}: SetupReviewStepProps) {
  const { t } = useTranslation("openhands");
  const declared = Object.entries(collectFields(setup));
  const rows: ReviewRow[] = [];

  if (
    automationName &&
    !declared.some(([name]) => isPreviewIdentityField(name))
  ) {
    rows.push({
      name: "name",
      label: t(I18nKey.AUTOMATIONS$NAME),
      value: automationName,
      kind: "name",
      parts: [automationName],
    });
  }

  for (const [name, field] of declared) {
    const parts = fieldValues(values[name]);
    rows.push({
      name,
      label: field.label,
      value: parts.join("\n") || t(I18nKey.SETUP$EMPTY_VALUE),
      kind: field.type,
      parts,
    });
  }

  const ordered = sortPreviewFields(
    rows,
    (row) => row.name,
    (row) => row.kind,
  );

  return (
    <dl className={automationPreviewListClassName} data-testid="setup-review">
      {ordered.map((row) => {
        const Icon = setupPreviewFieldIcon(
          row.name,
          row.kind === "name" ? "text" : (row.kind as SetupFieldType),
        );
        return (
          <AutomationPreviewField
            key={row.name}
            icon={<Icon className="size-3.5" />}
            label={row.label}
            value={row.value}
            chips={previewChipItems(row.name, row.kind, row.parts)}
            layout={
              row.value.includes("\n") || row.kind === "prompt"
                ? "stacked"
                : "inline"
            }
          />
        );
      })}
    </dl>
  );
}
