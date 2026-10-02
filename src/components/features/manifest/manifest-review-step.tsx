import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import {
  collectFields,
  fieldValues,
} from "#/manifests/manifest-local-validation";
import type { SetupBlock, SetupFormValues } from "#/manifests/types";

export interface SetupReviewStepProps {
  setup: SetupBlock;
  values: SetupFormValues;
  preflightStatus?: "passed" | "unsupported" | null;
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
  preflightStatus = null,
  selectedTrigger,
  selectedAction,
}: SetupReviewStepProps) {
  const { t } = useTranslation("openhands");

  return (
    <div className="flex flex-col gap-4" data-testid="setup-review">
      {preflightStatus === "passed" && (
        <p
          data-testid="setup-preflight-passed"
          className="rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-300"
        >
          {t(I18nKey.SETUP$PREFLIGHT_PASSED)}
        </p>
      )}
      {preflightStatus === "unsupported" && (
        <p
          data-testid="setup-preflight-unsupported"
          className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-200"
        >
          {t(I18nKey.SETUP$PREFLIGHT_UNSUPPORTED)}
        </p>
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
