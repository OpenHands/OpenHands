import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { getIntegrationCatalogEntry } from "@openhands/extensions/integrations";
import type { SetupPreflightOutcome } from "#/hooks/use-manifest-preflight";
import {
  collectFields,
  fieldValues,
} from "#/manifests/manifest-local-validation";
import type { SetupBlock, SetupFormValues } from "#/manifests/types";

export interface SetupReviewStepProps {
  setup: SetupBlock;
  values: SetupFormValues;
  preflightOutcome?: SetupPreflightOutcome | null;
  selectedTrigger?: string | null;
  selectedAction?: string | null;
}

/**
 * Stage 7 — the plain-language summary the user confirms.
 *
 * The last cheap moment to catch a wrong answer, and the last point at which
 * nothing has been created yet. A manifest declares no summary of its own: one
 * row per declared field, labelled the way the field was labelled, says the
 * same thing without asking every entry to restate it.
 */
export function SetupReviewStep({
  setup,
  values,
  preflightOutcome = null,
  selectedTrigger,
  selectedAction,
}: SetupReviewStepProps) {
  const { t } = useTranslation("openhands");
  const partialIntegrationNames =
    preflightOutcome?.status === "partial"
      ? preflightOutcome.integrationIds.map(
          (id) => getIntegrationCatalogEntry(id)?.name ?? id,
        )
      : [];

  return (
    <div className="flex flex-col gap-4" data-testid="setup-review">
      {preflightOutcome?.status === "passed" && (
        <p
          data-testid="setup-preflight-passed"
          className="rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-300"
        >
          {t(I18nKey.SETUP$PREFLIGHT_PASSED)}
        </p>
      )}
      {preflightOutcome?.status === "unsupported" && (
        <p
          data-testid="setup-preflight-unsupported"
          className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-200"
        >
          {t(I18nKey.SETUP$PREFLIGHT_UNSUPPORTED)}
        </p>
      )}
      {preflightOutcome?.status === "partial" && (
        <div
          data-testid="setup-preflight-partial"
          className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-200"
        >
          <p>{t(I18nKey.SETUP$PREFLIGHT_PARTIAL)}</p>
          {partialIntegrationNames.length > 0 && (
            <p data-testid="setup-preflight-partial-integrations">
              {partialIntegrationNames.join(", ")}
            </p>
          )}
        </div>
      )}
      <dl className="flex flex-col gap-3">
        {selectedAction &&
          setup.actions?.[selectedAction as keyof typeof setup.actions] && (
            <div className="flex flex-col gap-0.5">
              <dt className="text-xs text-muted">
                {t(I18nKey.SETUP$ACTION_LABEL)}
              </dt>
              <dd className="text-sm break-words">
                {
                  setup.actions[selectedAction as keyof typeof setup.actions]
                    ?.label
                }
              </dd>
            </div>
          )}
        {Object.entries(
          collectFields(setup, selectedTrigger, selectedAction),
        ).map(([name, field]) => (
          <div key={name} className="flex flex-col gap-0.5">
            <dt className="text-xs text-muted">{field.label}</dt>
            <dd className="text-sm break-words">
              {/* A field collecting several values reads as a list of them. */}
              {fieldValues(values[name]).join(", ") ||
                t(I18nKey.SETUP$EMPTY_VALUE)}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
