import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { collectFields } from "#/manifests/manifest-local-validation";
import type { SetupBlock, SetupFormValues } from "#/manifests/types";

export interface SetupReviewStepProps {
  setup: SetupBlock;
  values: SetupFormValues;
}

/**
 * Stage 7 — the plain-language summary the user confirms.
 *
 * The last cheap moment to catch a wrong answer, and the last point at which
 * nothing has been created yet. A manifest declares no summary of its own: one
 * row per declared field, labelled the way the field was labelled, says the
 * same thing without asking every entry to restate it.
 */
export function SetupReviewStep({ setup, values }: SetupReviewStepProps) {
  const { t } = useTranslation("openhands");

  return (
    <div className="flex flex-col gap-4" data-testid="setup-review">
      <dl className="flex flex-col gap-3">
        {Object.entries(collectFields(setup)).map(([name, field]) => (
          <div key={name} className="flex flex-col gap-0.5">
            <dt className="text-xs text-[var(--oh-muted)]">{field.label}</dt>
            <dd className="text-sm break-words">
              {(values[name] ?? "").trim() || t(I18nKey.SETUP$EMPTY_VALUE)}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
