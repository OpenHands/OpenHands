import React from "react";
import { useTranslation } from "react-i18next";
import type {
  CreateLoopTriggerPayload,
  LoopTrigger,
  LoopTriggerType,
} from "#/api/loop-service/loop-types";
import { LoopDefinitionCard } from "#/components/features/loops/loop-definition-card";
import { TriggerEventsFeed } from "#/components/features/loops/trigger-events-feed";
import { TriggerForm } from "#/components/features/loops/trigger-form";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import {
  useCreateLoopTrigger,
  useFireLoopTrigger,
  useLoopDefinitions,
  useLoopEvents,
  useLoopTriggers,
  useUpdateLoopTrigger,
} from "#/hooks/query/use-loops";
import { I18nKey } from "#/i18n/declaration";
import { ToggleSwitch } from "#/ui/toggle-switch";
import { Typography } from "#/ui/typography";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import {
  extensionModuleCardGridClassName,
  extensionModuleCardGridContainerClassName,
} from "#/utils/extension-module-card-classes";
import { cn } from "#/utils/utils";

function triggerTypeKey(triggerType: LoopTriggerType): I18nKey {
  if (triggerType === "scheduled") return I18nKey.LOOPS$TYPE_SCHEDULED;
  if (triggerType === "on_commit") return I18nKey.LOOPS$TYPE_ON_COMMIT;
  if (triggerType === "on_pr") return I18nKey.LOOPS$TYPE_ON_PR;
  return I18nKey.LOOPS$TYPE_MANUAL;
}

function TriggerRow({ trigger }: { trigger: LoopTrigger }) {
  const { t } = useTranslation("openhands");
  const updateTrigger = useUpdateLoopTrigger();
  const fireTrigger = useFireLoopTrigger();

  return (
    <li
      data-testid={`loop-trigger-${trigger.id}`}
      className="flex flex-wrap items-center justify-between gap-2 rounded-xl bg-base-secondary p-3"
    >
      <span className="text-sm text-white">
        {t(triggerTypeKey(trigger.trigger_type))}
      </span>
      <span data-testid={`loop-trigger-enabled-${trigger.id}`}>
        <ToggleSwitch
          enabled={trigger.enabled}
          label={t(I18nKey.LOOPS$ENABLE)}
          onToggle={() =>
            updateTrigger.mutate({
              triggerId: trigger.id,
              payload: { enabled: !trigger.enabled },
            })
          }
        />
      </span>
      <BrandButton
        type="button"
        variant="secondary"
        testId={`loop-trigger-run-now-${trigger.id}`}
        onClick={() =>
          fireTrigger.mutate(trigger.id, {
            onError: () => displayErrorToast(t(I18nKey.ERROR$GENERIC)),
          })
        }
      >
        {t(I18nKey.LOOPS$RUN_NOW)}
      </BrandButton>
      <span className="text-xs text-tertiary-light">
        {t(I18nKey.LOOPS$LAST_FIRED)}
        <span className="ml-1 text-white">
          {trigger.last_fired_at ?? t(I18nKey.LOOPS$NEVER)}
        </span>
      </span>
    </li>
  );
}

export function LoopsOverview() {
  const { t } = useTranslation("openhands");
  const definitionsQuery = useLoopDefinitions();
  const triggersQuery = useLoopTriggers();
  const eventsQuery = useLoopEvents();
  const createTrigger = useCreateLoopTrigger();
  const [formOpen, setFormOpen] = React.useState(false);
  const definitions = definitionsQuery.data ?? [];
  const triggers = triggersQuery.data ?? [];
  const events = eventsQuery.data ?? [];
  const runTriggerByDefinition = React.useMemo(() => {
    const map = new Map<string, string>();
    for (const trigger of triggers) {
      if (!trigger.enabled || map.has(trigger.loop_definition_id)) continue;
      map.set(trigger.loop_definition_id, trigger.id);
    }
    return map;
  }, [triggers]);

  return (
    <div data-testid="loops-overview" className="flex flex-col gap-8">
      <section>
        <Typography.H3>{t(I18nKey.LOOPS$DEFINITIONS)}</Typography.H3>
        <div className={cn("mt-3", extensionModuleCardGridContainerClassName)}>
          {definitions.length === 0 ? (
            <p className="text-sm text-tertiary-light">
              {t(I18nKey.LOOPS$EMPTY_DEFINITIONS)}
            </p>
          ) : (
            <div className={extensionModuleCardGridClassName}>
              {definitions.map((definition) => (
                <LoopDefinitionCard
                  key={definition.id}
                  definition={definition}
                  runTriggerId={
                    runTriggerByDefinition.get(definition.id) ?? null
                  }
                />
              ))}
            </div>
          )}
        </div>
      </section>

      <section>
        <div className="mb-3 flex items-center justify-between gap-2">
          <Typography.H3>{t(I18nKey.LOOPS$TRIGGERS)}</Typography.H3>
          <BrandButton
            type="button"
            variant="primary"
            testId="loop-create-trigger"
            onClick={() => setFormOpen(true)}
          >
            {t(I18nKey.LOOPS$CREATE_TRIGGER)}
          </BrandButton>
        </div>
        {triggers.length === 0 ? (
          <p
            data-testid="loop-triggers-empty"
            className="text-sm text-tertiary-light"
          >
            {t(I18nKey.LOOPS$EMPTY_TRIGGERS)}
          </p>
        ) : (
          <ul className="flex flex-col gap-2">
            {triggers.map((trigger) => (
              <TriggerRow key={trigger.id} trigger={trigger} />
            ))}
          </ul>
        )}
      </section>

      <section>
        <Typography.H3>{t(I18nKey.LOOPS$EVENTS)}</Typography.H3>
        <div className="mt-3">
          <TriggerEventsFeed events={events} triggers={triggers} />
        </div>
      </section>

      <ApiKeyModalBase
        isOpen={formOpen}
        title={t(I18nKey.LOOPS$CREATE_TRIGGER)}
        onClose={() => setFormOpen(false)}
        footer={null}
      >
        <TriggerForm
          definitions={definitions}
          onCancel={() => setFormOpen(false)}
          onSubmit={(payload: CreateLoopTriggerPayload) => {
            createTrigger.mutate(payload, {
              onSuccess: () => setFormOpen(false),
              onError: () => displayErrorToast(t(I18nKey.ERROR$GENERIC)),
            });
          }}
        />
      </ApiKeyModalBase>
    </div>
  );
}
