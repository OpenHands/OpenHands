import React from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router";
import { Plus, Trash2 } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { BackNavButton } from "#/components/shared/buttons/back-nav-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { BrandButton } from "#/components/features/settings/brand-button";
import { formControlMultilineFieldClassName } from "#/utils/form-control-classes";
import { Typography } from "#/ui/typography";
import { useCreateExperimentAutomation } from "#/hooks/query/use-experiment-automation";
import { getApiErrorMessage } from "#/utils/api-error-message";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import type { ExperimentVariant } from "#/types/experiment";

export const handle = { hideTitle: true };

let variantKeySeed = 0;
function nextVariantKey() {
  variantKeySeed += 1;
  return `variant-${variantKeySeed}`;
}

interface VariantDraft {
  key: string;
  name: string;
  weight: string;
  model: string;
  pluginSource: string;
}

function emptyVariant(): VariantDraft {
  return {
    key: nextVariantKey(),
    name: "",
    weight: "1",
    model: "",
    pluginSource: "",
  };
}

export function AutomationExperimentSetupScreen() {
  const { t } = useTranslation("openhands");
  const navigate = useNavigate();
  const { mutate: createExperiment, isPending } =
    useCreateExperimentAutomation();

  const [name, setName] = React.useState("");
  const [prompt, setPrompt] = React.useState("");
  const [schedule, setSchedule] = React.useState("0 9 * * 1");
  const [experimentId, setExperimentId] = React.useState("");
  const [variants, setVariants] = React.useState<VariantDraft[]>([
    emptyVariant(),
    emptyVariant(),
  ]);
  const [error, setError] = React.useState<string | null>(null);

  const updateVariant = (key: string, patch: Partial<VariantDraft>) => {
    setVariants((current) =>
      current.map((v) => (v.key === key ? { ...v, ...patch } : v)),
    );
  };

  const addVariant = () =>
    setVariants((current) => [...current, emptyVariant()]);
  const removeVariant = (key: string) =>
    setVariants((current) => current.filter((v) => v.key !== key));

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    if (!name.trim() || !prompt.trim() || !experimentId.trim()) {
      setError(t(I18nKey.AUTOMATIONS$EXPERIMENT$MISSING_REQUIRED));
      return;
    }
    if (variants.length < 2) {
      setError(t(I18nKey.AUTOMATIONS$EXPERIMENT$NEED_TWO_VARIANTS));
      return;
    }
    const names = variants.map((v) => v.name.trim());
    if (names.some((n) => !n) || new Set(names).size !== names.length) {
      setError(t(I18nKey.AUTOMATIONS$EXPERIMENT$VARIANT_NAMES_INVALID));
      return;
    }
    if (variants.some((v) => !v.pluginSource.trim())) {
      setError(t(I18nKey.AUTOMATIONS$EXPERIMENT$VARIANT_PLUGIN_REQUIRED));
      return;
    }

    const payloadVariants: ExperimentVariant[] = variants.map((v) => ({
      name: v.name.trim(),
      weight: Math.max(1, Number(v.weight) || 1),
      ...(v.model.trim() && { model: v.model.trim() }),
      plugins: [{ source: v.pluginSource.trim() }],
    }));

    createExperiment(
      {
        name: name.trim(),
        prompt: prompt.trim(),
        trigger: { type: "cron", schedule: schedule.trim() },
        variants: payloadVariants,
        experiment_id: experimentId.trim(),
      },
      {
        onSuccess: (automation) => {
          displaySuccessToast(t(I18nKey.AUTOMATIONS$EXPERIMENT$CREATED));
          navigate(`/automations/${automation.id}`);
        },
        onError: (err) => {
          setError(
            getApiErrorMessage(err, t(I18nKey.AUTOMATIONS$EXPERIMENT$ERROR)),
          );
          displayErrorToast(
            getApiErrorMessage(err, t(I18nKey.AUTOMATIONS$EXPERIMENT$ERROR)),
          );
        },
      },
    );
  };

  return (
    <div
      data-testid="automation-experiment-setup-screen"
      className="flex flex-col gap-6"
    >
      <div className="flex flex-col gap-2">
        <BackNavButton
          testId="back-to-automations"
          onClick={() => navigate("/automations")}
        >
          {t(I18nKey.BUTTON$BACK)}
        </BackNavButton>
        <Typography.H2>{t(I18nKey.AUTOMATIONS$EXPERIMENT$TITLE)}</Typography.H2>
        <p className="text-sm leading-5 text-tertiary-light">
          {t(I18nKey.AUTOMATIONS$EXPERIMENT$SUBLINE)}
        </p>
      </div>

      <form
        data-testid="experiment-setup-form"
        onSubmit={handleSubmit}
        className="flex flex-col items-start gap-6"
      >
        <SettingsInput
          testId="experiment-name-input"
          name="experiment-name"
          type="text"
          label={t(I18nKey.SETTINGS$NAME)}
          className="w-full min-w-0"
          required
          value={name}
          onChange={setName}
        />

        <label className="flex w-full min-w-0 flex-col gap-2.5">
          <span className="text-sm">
            {t(I18nKey.AUTOMATIONS$EXPERIMENT$PROMPT)}
          </span>
          <textarea
            data-testid="experiment-prompt-input"
            required
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className={formControlMultilineFieldClassName}
            rows={4}
          />
        </label>

        <SettingsInput
          testId="experiment-schedule-input"
          name="experiment-schedule"
          type="text"
          label={t(I18nKey.AUTOMATIONS$DETAIL$SCHEDULE)}
          className="w-full min-w-0"
          required
          value={schedule}
          onChange={setSchedule}
          placeholder={t(I18nKey.AUTOMATIONS$EXPERIMENT$SCHEDULE_PLACEHOLDER)}
        />

        <SettingsInput
          testId="experiment-id-input"
          name="experiment-id"
          type="text"
          label={t(I18nKey.AUTOMATIONS$EXPERIMENT$EXPERIMENT_ID)}
          className="w-full min-w-0"
          required
          value={experimentId}
          onChange={setExperimentId}
        />

        <div className="flex w-full min-w-0 flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-sm">
              {t(I18nKey.AUTOMATIONS$EXPERIMENT$VARIANTS)}
            </span>
            <button
              type="button"
              data-testid="add-variant-button"
              onClick={addVariant}
              className="flex items-center gap-1 text-xs text-[var(--oh-primary)] hover:underline"
            >
              <Plus className="size-3.5" aria-hidden />
              {t(I18nKey.AUTOMATIONS$EXPERIMENT$ADD_VARIANT)}
            </button>
          </div>

          {variants.map((variant, index) => (
            <div
              key={variant.key}
              data-testid="variant-row"
              className="flex flex-col gap-2 rounded-lg border border-[var(--oh-border)] p-3"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted">
                  {t(I18nKey.AUTOMATIONS$EXPERIMENT$VARIANT_N, {
                    n: index + 1,
                  })}
                </span>
                {variants.length > 2 && (
                  <button
                    type="button"
                    data-testid="remove-variant-button"
                    onClick={() => removeVariant(variant.key)}
                    aria-label={t(
                      I18nKey.AUTOMATIONS$EXPERIMENT$REMOVE_VARIANT,
                    )}
                    className="text-muted hover:text-red-500"
                  >
                    <Trash2 className="size-3.5" aria-hidden />
                  </button>
                )}
              </div>
              <div className="grid grid-cols-2 gap-3">
                <SettingsInput
                  testId={`variant-name-input-${index}`}
                  name={`variant-name-${variant.key}`}
                  type="text"
                  label={t(I18nKey.SETTINGS$NAME)}
                  value={variant.name}
                  onChange={(v) => updateVariant(variant.key, { name: v })}
                  required
                />
                <SettingsInput
                  testId={`variant-weight-input-${index}`}
                  name={`variant-weight-${variant.key}`}
                  type="text"
                  label={t(I18nKey.AUTOMATIONS$EXPERIMENT$WEIGHT)}
                  value={variant.weight}
                  onChange={(v) => updateVariant(variant.key, { weight: v })}
                />
                <SettingsInput
                  testId={`variant-model-input-${index}`}
                  name={`variant-model-${variant.key}`}
                  type="text"
                  label={t(I18nKey.AUTOMATIONS$DETAIL$MODEL)}
                  value={variant.model}
                  onChange={(v) => updateVariant(variant.key, { model: v })}
                />
                <SettingsInput
                  testId={`variant-plugin-input-${index}`}
                  name={`variant-plugin-${variant.key}`}
                  type="text"
                  label={t(I18nKey.AUTOMATIONS$EXPERIMENT$PLUGIN_SOURCE)}
                  value={variant.pluginSource}
                  onChange={(v) =>
                    updateVariant(variant.key, { pluginSource: v })
                  }
                  placeholder={t(
                    I18nKey.AUTOMATIONS$EXPERIMENT$PLUGIN_SOURCE_PLACEHOLDER,
                  )}
                  required
                />
              </div>
            </div>
          ))}
        </div>

        {error && <p className="text-sm text-red-500">{error}</p>}

        <div className="flex items-center gap-4">
          <BrandButton
            testId="experiment-cancel-button"
            type="button"
            variant="secondary"
            onClick={() => navigate("/automations")}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          <BrandButton
            testId="experiment-submit-button"
            type="submit"
            variant="primary"
            isDisabled={isPending}
          >
            {t(I18nKey.AUTOMATIONS$EXPERIMENT$CREATE)}
          </BrandButton>
        </div>
      </form>
    </div>
  );
}

export default AutomationExperimentSetupScreen;
