import { useTranslation } from "react-i18next";
import { loopRunPath } from "#/api/loop-service/loop-constants";
import type { LoopTrigger, TriggerEvent } from "#/api/loop-service/loop-types";
import { I18nKey } from "#/i18n/declaration";
import { useNavigation } from "#/context/navigation-context";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import { cn } from "#/utils/utils";

const STATUS_KEY: Record<string, I18nKey> = {
  fired: I18nKey.LOOPS$STATUS_FIRED,
  skipped: I18nKey.LOOPS$STATUS_SKIPPED,
  error: I18nKey.LOOPS$STATUS_ERROR,
};

const TYPE_KEY: Record<string, I18nKey> = {
  scheduled: I18nKey.LOOPS$TYPE_SCHEDULED,
  on_commit: I18nKey.LOOPS$TYPE_ON_COMMIT,
  on_pr: I18nKey.LOOPS$TYPE_ON_PR,
  manual: I18nKey.LOOPS$TYPE_MANUAL,
};

export interface TriggerEventsFeedProps {
  events: TriggerEvent[];
  triggers: LoopTrigger[];
}

export function TriggerEventsFeed({
  events,
  triggers,
}: TriggerEventsFeedProps) {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const triggerById = new Map(triggers.map((trigger) => [trigger.id, trigger]));

  if (events.length === 0) {
    return (
      <p
        data-testid="loop-events-empty"
        className="text-sm text-tertiary-light"
      >
        {t(I18nKey.LOOPS$EMPTY_EVENTS)}
      </p>
    );
  }

  return (
    <ul data-testid="loop-events-feed" className="flex flex-col gap-2">
      {events.map((event) => {
        const trigger = triggerById.get(event.trigger_id);
        return (
          <li
            key={event.id}
            data-testid={`loop-event-${event.id}`}
            className="flex flex-wrap items-center justify-between gap-2 rounded-xl bg-base-secondary p-3"
          >
            <span
              className={cn(extensionModuleCardPillClassName, "text-white")}
            >
              {t(
                TYPE_KEY[trigger?.trigger_type ?? "manual"] ??
                  I18nKey.LOOPS$TYPE_MANUAL,
              )}
            </span>
            <span
              data-testid={`loop-event-status-${event.id}`}
              className={cn(extensionModuleCardPillClassName, "text-white")}
            >
              {t(STATUS_KEY[event.status] ?? I18nKey.LOOPS$STATUS_FIRED)}
            </span>
            <time className="text-xs text-tertiary-light">
              {event.fired_at}
            </time>
            {event.loop_run_id ? (
              <button
                type="button"
                data-testid={`loop-event-run-${event.id}`}
                className="text-sm text-primary"
                onClick={() => navigate(loopRunPath(event.loop_run_id!))}
              >
                {event.loop_run_id}
              </button>
            ) : (
              <span className="text-xs text-tertiary-light">
                {t(I18nKey.LOOPS$NEVER)}
              </span>
            )}
          </li>
        );
      })}
    </ul>
  );
}
