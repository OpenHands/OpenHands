import React from "react";
import { useTranslation } from "react-i18next";
import type {
  CreateLoopTriggerPayload,
  LoopDefinition,
  LoopScheduleType,
  LoopTrigger,
  LoopTriggerType,
} from "#/api/loop-service/loop-types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { I18nKey } from "#/i18n/declaration";

export interface TriggerFormValues {
  project_id: string;
  loop_definition_id: string;
  trigger_type: LoopTriggerType;
  schedule_type: LoopScheduleType | "";
  cron_expr: string;
  interval_seconds: string;
}

export interface TriggerFormProps {
  definitions: LoopDefinition[];
  initial?: LoopTrigger | null;
  projectId?: string;
  onSubmit: (payload: CreateLoopTriggerPayload) => void;
  onCancel: () => void;
}

const DEFAULT_VALUES: TriggerFormValues = {
  project_id: "proj-1",
  loop_definition_id: "",
  trigger_type: "manual",
  schedule_type: "",
  cron_expr: "",
  interval_seconds: "",
};

export function validateTriggerForm(values: TriggerFormValues): I18nKey | null {
  if (values.trigger_type !== "scheduled") {
    return null;
  }
  if (values.schedule_type === "cron" && values.cron_expr.trim()) {
    return null;
  }
  if (
    values.schedule_type === "interval" &&
    Number(values.interval_seconds) >= 1
  ) {
    return null;
  }
  return I18nKey.LOOPS$VALIDATION_SCHEDULE;
}

export function TriggerForm({
  definitions,
  initial,
  projectId,
  onSubmit,
  onCancel,
}: TriggerFormProps) {
  const { t } = useTranslation("openhands");
  const [values, setValues] = React.useState<TriggerFormValues>(() =>
    initial
      ? {
          project_id: initial.project_id,
          loop_definition_id: initial.loop_definition_id,
          trigger_type: initial.trigger_type,
          schedule_type: initial.schedule_type ?? "",
          cron_expr: initial.cron_expr ?? "",
          interval_seconds:
            initial.interval_seconds != null
              ? String(initial.interval_seconds)
              : "",
        }
      : {
          ...DEFAULT_VALUES,
          project_id: projectId || DEFAULT_VALUES.project_id,
          loop_definition_id: definitions[0]?.id ?? "",
        },
  );
  const [error, setError] = React.useState<I18nKey | null>(null);

  const scheduled = values.trigger_type === "scheduled";

  return (
    <form
      data-testid="loop-trigger-form"
      className="flex flex-col gap-3"
      onSubmit={(event) => {
        event.preventDefault();
        const validation = validateTriggerForm(values);
        setError(validation);
        if (validation) {
          return;
        }
        onSubmit({
          project_id: values.project_id,
          loop_definition_id: values.loop_definition_id,
          trigger_type: values.trigger_type,
          schedule_type: scheduled
            ? (values.schedule_type as LoopScheduleType)
            : null,
          cron_expr:
            scheduled && values.schedule_type === "cron"
              ? values.cron_expr.trim()
              : null,
          interval_seconds:
            scheduled && values.schedule_type === "interval"
              ? Number(values.interval_seconds)
              : null,
        });
      }}
    >
      <SettingsDropdownInput
        testId="loop-trigger-type"
        name="loop-trigger-type"
        label={t(I18nKey.LOOPS$TYPE)}
        selectedKey={values.trigger_type}
        items={[
          { key: "scheduled", label: t(I18nKey.LOOPS$TYPE_SCHEDULED) },
          { key: "on_commit", label: t(I18nKey.LOOPS$TYPE_ON_COMMIT) },
          { key: "on_pr", label: t(I18nKey.LOOPS$TYPE_ON_PR) },
          { key: "manual", label: t(I18nKey.LOOPS$TYPE_MANUAL) },
        ]}
        onSelectionChange={(key) =>
          setValues((current) => ({
            ...current,
            trigger_type: (key ? String(key) : "manual") as LoopTriggerType,
            schedule_type:
              key === "scheduled" ? current.schedule_type || "interval" : "",
          }))
        }
      />
      <SettingsDropdownInput
        testId="loop-trigger-definition"
        name="loop-trigger-definition"
        label={t(I18nKey.LOOPS$LOOP_DEFINITION)}
        selectedKey={values.loop_definition_id}
        items={definitions.map((definition) => ({
          key: definition.id,
          label: definition.name,
        }))}
        onSelectionChange={(key) =>
          setValues((current) => ({
            ...current,
            loop_definition_id: key ? String(key) : "",
          }))
        }
      />
      {scheduled ? (
        <>
          <SettingsDropdownInput
            testId="loop-trigger-schedule-type"
            name="loop-trigger-schedule-type"
            label={t(I18nKey.LOOPS$SCHEDULE)}
            selectedKey={values.schedule_type}
            items={[
              { key: "cron", label: t(I18nKey.LOOPS$SCHEDULE_CRON) },
              { key: "interval", label: t(I18nKey.LOOPS$SCHEDULE_INTERVAL) },
            ]}
            onSelectionChange={(key) =>
              setValues((current) => ({
                ...current,
                schedule_type: (key ? String(key) : "") as
                  | LoopScheduleType
                  | "",
              }))
            }
          />
          {values.schedule_type === "cron" ? (
            <SettingsInput
              testId="loop-trigger-cron"
              name="loop-trigger-cron"
              type="text"
              label={t(I18nKey.LOOPS$CRON)}
              value={values.cron_expr}
              onChange={(value) =>
                setValues((current) => ({ ...current, cron_expr: value }))
              }
            />
          ) : null}
          {values.schedule_type === "interval" ? (
            <SettingsInput
              testId="loop-trigger-interval"
              name="loop-trigger-interval"
              type="number"
              min={1}
              label={t(I18nKey.LOOPS$INTERVAL_SECONDS)}
              value={values.interval_seconds}
              onChange={(value) =>
                setValues((current) => ({
                  ...current,
                  interval_seconds: value,
                }))
              }
            />
          ) : null}
        </>
      ) : null}
      {error ? (
        <p
          data-testid="loop-trigger-schedule-error"
          className="text-sm text-red-400"
        >
          {t(error)}
        </p>
      ) : null}
      <div className="flex justify-end gap-2">
        <BrandButton
          type="button"
          variant="tertiary"
          testId="loop-trigger-cancel"
          onClick={onCancel}
        >
          {t(I18nKey.BUTTON$CANCEL)}
        </BrandButton>
        <BrandButton type="submit" variant="primary" testId="loop-trigger-save">
          {t(I18nKey.BUTTON$SAVE)}
        </BrandButton>
      </div>
    </form>
  );
}
