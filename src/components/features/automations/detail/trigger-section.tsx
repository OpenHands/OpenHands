import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Zap } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";
import CalendarIcon from "#/icons/calendar.svg?react";
import ClockIcon from "#/icons/clock.svg?react";
import CodeTagIcon from "#/icons/code-tag.svg?react";
import { describeTrigger } from "#/utils/automation-trigger-description";
import { SectionCard } from "./section-card";
import { ConfigField } from "./config-field";

const FILTER_TRUNCATE_LENGTH = 60;

function FilterExpression({ filter }: { filter: string }) {
  const { t } = useTranslation("openhands");
  const [expanded, setExpanded] = useState(false);
  const isLong = filter.length > FILTER_TRUNCATE_LENGTH;

  return (
    <div className="flex flex-col gap-1">
      <span className="font-mono break-all">
        {isLong && !expanded
          ? `${filter.slice(0, FILTER_TRUNCATE_LENGTH)}…`
          : filter}
      </span>
      {isLong && (
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          aria-expanded={expanded}
          className="text-xs text-muted hover:text-content self-start"
        >
          {expanded
            ? t(I18nKey.SETTINGS$SKILLS_SHOW_LESS)
            : t(I18nKey.SETTINGS$SKILLS_SHOW_MORE)}
        </button>
      )}
    </div>
  );
}

/**
 * What starts the automation, worded by the same formatter the list card
 * uses, with the exact cron or filter underneath for whoever needs it.
 */
export function TriggerSection({ automation }: { automation: Automation }) {
  const { t, i18n } = useTranslation("openhands");
  const { trigger } = automation;
  const isEvent = trigger.type === "event";
  const timezone = automation.timezone ?? trigger.timezone;
  const summary = describeTrigger(trigger, t, i18n.language);
  // A cron with no plainer wording is its own summary; don't repeat it.
  const cron = trigger.schedule?.trim() !== summary ? trigger.schedule : null;

  return (
    <SectionCard
      testId="automation-trigger-section"
      icon={
        isEvent ? (
          <Zap className="size-4" aria-hidden="true" />
        ) : (
          <CalendarIcon className="size-4" />
        )
      }
      title={t(I18nKey.AUTOMATIONS$DETAIL$TRIGGER)}
      meta={
        isEvent
          ? t(I18nKey.AUTOMATIONS$DETAIL$TRIGGER_EVENT)
          : t(I18nKey.AUTOMATIONS$DETAIL$TRIGGER_SCHEDULE)
      }
    >
      <p
        data-testid="automation-trigger-summary"
        className="break-words text-base text-content"
      >
        {summary}
      </p>

      {!isEvent && (timezone || cron) ? (
        <div className="mt-5 grid grid-cols-2 gap-x-4 gap-y-5">
          {timezone ? (
            <ConfigField
              icon={<ClockIcon className="size-3.5" />}
              label={t(I18nKey.AUTOMATIONS$TIMEZONE)}
            >
              {timezone}
            </ConfigField>
          ) : null}
          {cron ? (
            <ConfigField
              icon={<CodeTagIcon className="size-3.5" />}
              label={t(I18nKey.AUTOMATIONS$DETAIL$CRON_EXPRESSION)}
            >
              <span className="font-mono">{cron}</span>
            </ConfigField>
          ) : null}
        </div>
      ) : null}

      {isEvent && trigger.filter ? (
        <div className="mt-5">
          <ConfigField
            icon={<CodeTagIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$EVENT_FILTER)}
          >
            <FilterExpression filter={trigger.filter} />
          </ConfigField>
        </div>
      ) : null}
    </SectionCard>
  );
}
