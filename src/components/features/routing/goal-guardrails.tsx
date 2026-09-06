import React from "react";
import { useTranslation } from "react-i18next";
import type {
  RoutingConfig,
  RoutingMode,
} from "#/api/routing-service/routing-types";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { I18nKey } from "#/i18n/declaration";
import { ToggleSwitch } from "#/ui/toggle-switch";

export interface GoalGuardrailsProps {
  config: RoutingConfig;
  onChange: (patch: Partial<RoutingConfig>) => void;
}

export function GoalGuardrails({ config, onChange }: GoalGuardrailsProps) {
  const { t } = useTranslation("openhands");
  const strict = config.mode === "strict";
  return (
    <section
      data-testid="routing-goal-guardrails"
      className="flex flex-col gap-3"
    >
      <h3 className="text-sm font-medium text-white">
        {t(I18nKey.ROUTING$GOALS)}
      </h3>
      <ToggleSwitch
        enabled={strict}
        label={t(I18nKey.ROUTING$MODE)}
        onToggle={() => {
          const mode: RoutingMode = strict ? "warn" : "strict";
          onChange({ mode });
        }}
      />
      <span
        data-testid="routing-mode-label"
        className="text-xs text-[var(--oh-muted)]"
      >
        {strict ? t(I18nKey.ROUTING$STRICT) : t(I18nKey.ROUTING$WARN)}
      </span>
      <ToggleSwitch
        enabled={config.guardrails.forbid_training_retention}
        label={t(I18nKey.ROUTING$FORBID_RETENTION)}
        onToggle={() =>
          onChange({
            guardrails: {
              ...config.guardrails,
              forbid_training_retention:
                !config.guardrails.forbid_training_retention,
            },
          })
        }
      />
      <ToggleSwitch
        enabled={config.guardrails.forbid_watermarking}
        label={t(I18nKey.ROUTING$FORBID_WATERMARK)}
        onToggle={() =>
          onChange({
            guardrails: {
              ...config.guardrails,
              forbid_watermarking: !config.guardrails.forbid_watermarking,
            },
          })
        }
      />
      <SettingsInput
        testId="routing-max-cost"
        name="routing-max-cost"
        label={t(I18nKey.ROUTING$MAX_COST)}
        type="number"
        value={
          config.guardrails.max_cost_usd_per_task == null
            ? ""
            : String(config.guardrails.max_cost_usd_per_task)
        }
        onChange={(value) =>
          onChange({
            guardrails: {
              ...config.guardrails,
              max_cost_usd_per_task: value === "" ? null : Number(value),
            },
          })
        }
      />
      <SettingsInput
        testId="routing-max-latency"
        name="routing-max-latency"
        label={t(I18nKey.ROUTING$MAX_LATENCY)}
        type="number"
        value={
          config.guardrails.max_latency_s == null
            ? ""
            : String(config.guardrails.max_latency_s)
        }
        onChange={(value) =>
          onChange({
            guardrails: {
              ...config.guardrails,
              max_latency_s: value === "" ? null : Number(value),
            },
          })
        }
      />
    </section>
  );
}
