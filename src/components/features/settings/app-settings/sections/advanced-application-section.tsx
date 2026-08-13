import React from "react";
import { useTranslation } from "react-i18next";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useSettings } from "#/hooks/query/use-settings";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { I18nKey } from "#/i18n/declaration";
import { parseMaxBudgetPerTask } from "#/utils/settings-utils";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

/**
 * Advanced Application settings that only exist on cloud/enterprise backends:
 * a per-task budget cap, solvability analysis, and proactive GitHub task
 * suggestions. Registered with a `when: ctx => ctx.backendKind === "cloud"`
 * predicate rather than an inline `if (isCloud)` in the page host.
 *
 * Section-owned save — it persists only these three fields, so the host never
 * needs to know they exist. This is the first backend-specific section to use
 * the registry's `when` gate (issue #16596): it closes part of the concrete
 * enterprise-parity gap the registry was built to close.
 */
export function AdvancedApplicationSection() {
  const { t } = useTranslation("openhands");
  const { mutate: saveSettings, isPending } = useSaveSettings();
  const { data: settings } = useSettings();

  const [maxBudgetHasChanged, setMaxBudgetHasChanged] = React.useState(false);
  const [solvabilityHasChanged, setSolvabilityHasChanged] =
    React.useState(false);
  const [proactiveHasChanged, setProactiveHasChanged] = React.useState(false);

  if (!settings) return null;

  const storedMaxBudget =
    settings.max_budget_per_task === null ||
    settings.max_budget_per_task === undefined
      ? ""
      : String(settings.max_budget_per_task);

  const formAction = (formData: FormData) => {
    const maxBudget = parseMaxBudgetPerTask(
      formData.get("max-budget-per-task-input")?.toString() ?? "",
    );
    const enableSolvability =
      formData.get("enable-solvability-analysis-switch")?.toString() === "on";
    const enableProactive =
      formData
        .get("enable-proactive-conversation-starters-switch")
        ?.toString() === "on";

    saveSettings(
      {
        max_budget_per_task: maxBudget,
        enable_solvability_analysis: enableSolvability,
        enable_proactive_conversation_starters: enableProactive,
      },
      {
        onSuccess: () => displaySuccessToast(t(I18nKey.SETTINGS$SAVED)),
        onError: (error) => {
          const errorMessage = retrieveAxiosErrorMessage(error);
          displayErrorToast(errorMessage || t(I18nKey.ERROR$GENERIC));
        },
        onSettled: () => {
          setMaxBudgetHasChanged(false);
          setSolvabilityHasChanged(false);
          setProactiveHasChanged(false);
        },
      },
    );
  };

  const checkIfMaxBudgetHasChanged = (value: string) => {
    setMaxBudgetHasChanged(value !== storedMaxBudget);
  };

  const checkIfSolvabilityHasChanged = (checked: boolean) => {
    setSolvabilityHasChanged(
      checked !== !!settings.enable_solvability_analysis,
    );
  };

  const checkIfProactiveHasChanged = (checked: boolean) => {
    setProactiveHasChanged(
      checked !== !!settings.enable_proactive_conversation_starters,
    );
  };

  const formIsClean =
    !maxBudgetHasChanged && !solvabilityHasChanged && !proactiveHasChanged;

  return (
    <form
      data-testid="app-settings-advanced-section"
      action={formAction}
      className="flex flex-col gap-6"
    >
      <SettingsInput
        testId="max-budget-per-task-input"
        name="max-budget-per-task-input"
        type="number"
        label={t(I18nKey.SETTINGS$MAX_BUDGET_PER_TASK)}
        defaultValue={storedMaxBudget}
        min={1}
        step={1}
        showOptionalTag
        onChange={checkIfMaxBudgetHasChanged}
        className="w-full max-w-[680px]"
      />

      <SettingsSwitch
        testId="enable-solvability-analysis-switch"
        name="enable-solvability-analysis-switch"
        defaultIsToggled={!!settings.enable_solvability_analysis}
        onToggle={checkIfSolvabilityHasChanged}
      >
        {t(I18nKey.SETTINGS$SOLVABILITY_ANALYSIS)}
      </SettingsSwitch>

      <SettingsSwitch
        testId="enable-proactive-conversation-starters-switch"
        name="enable-proactive-conversation-starters-switch"
        defaultIsToggled={!!settings.enable_proactive_conversation_starters}
        onToggle={checkIfProactiveHasChanged}
      >
        {t(I18nKey.SETTINGS$PROACTIVE_CONVERSATION_STARTERS)}
      </SettingsSwitch>

      <div className="flex justify-start">
        <BrandButton
          testId="app-settings-advanced-submit"
          variant="primary"
          type="submit"
          isDisabled={isPending || formIsClean}
        >
          {!isPending && t(I18nKey.SETTINGS$SAVE_CHANGES)}
          {isPending && t(I18nKey.SETTINGS$SAVING)}
        </BrandButton>
      </div>
    </form>
  );
}
