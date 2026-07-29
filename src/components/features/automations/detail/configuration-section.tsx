import { useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";
import CogIcon from "#/icons/cog.svg?react";
import GitBranchIcon from "#/icons/git-branch.svg?react";
import CheckCircleIcon from "#/icons/check-circle.svg?react";
import CalendarIcon from "#/icons/calendar.svg?react";
import SparkleIcon from "#/icons/sparkle.svg?react";
import BellIcon from "#/icons/bell.svg?react";
import GlobeIcon from "#/icons/globe.svg?react";
import CodeTagIcon from "#/icons/code-tag.svg?react";
import LinkExternalIcon from "#/icons/link-external.svg?react";
import { formatEventOn } from "#/utils/automation-schedule";
import { SectionCard } from "./section-card";
import { ConfigField } from "./config-field";
import { BranchBadge } from "./branch-badge";

interface ConfigurationSectionProps {
  automation: Automation;
}

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

export function ConfigurationSection({
  automation,
}: ConfigurationSectionProps) {
  const { t } = useTranslation("openhands");
  const isEvent = automation.trigger.type === "event";

  let scheduleDisplay = automation.trigger.schedule ?? "";
  if (automation.trigger.schedule_human) {
    scheduleDisplay = automation.timezone
      ? `${automation.trigger.schedule_human} (${automation.timezone})`
      : automation.trigger.schedule_human;
  }

  const triggerDisplay = isEvent
    ? t(I18nKey.AUTOMATIONS$DETAIL$TRIGGER_EVENT)
    : t(I18nKey.AUTOMATIONS$DETAIL$TRIGGER_SCHEDULE);

  return (
    <SectionCard
      icon={<CogIcon className="size-4" />}
      title={t(I18nKey.AUTOMATIONS$DETAIL$CONFIGURATION)}
    >
      <div className="grid grid-cols-2 gap-x-4 gap-y-5">
        {automation.repository && (
          <ConfigField
            icon={<GitBranchIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$REPOSITORIES)}
          >
            <span className="flex items-center gap-1">
              {automation.repository}
              {automation.branch && <BranchBadge branch={automation.branch} />}
            </span>
          </ConfigField>
        )}

        <ConfigField
          icon={<CheckCircleIcon className="size-3.5" />}
          label={t(I18nKey.AUTOMATIONS$DETAIL$TRIGGER)}
        >
          {triggerDisplay}
        </ConfigField>

        {!isEvent && (
          <ConfigField
            icon={<CalendarIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$SCHEDULE)}
          >
            {scheduleDisplay}
          </ConfigField>
        )}

        {isEvent && automation.trigger.source && (
          <ConfigField
            icon={<GlobeIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$EVENT_SOURCE)}
          >
            {automation.trigger.source}
          </ConfigField>
        )}

        {isEvent && automation.trigger.on && (
          <ConfigField
            icon={<LinkExternalIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$EVENT_TYPE)}
          >
            {formatEventOn(automation.trigger.on)}
          </ConfigField>
        )}

        {isEvent && automation.trigger.filter && (
          <ConfigField
            icon={<CodeTagIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$EVENT_FILTER)}
          >
            <FilterExpression filter={automation.trigger.filter} />
          </ConfigField>
        )}

        <ConfigField
          icon={<SparkleIcon className="size-3.5" />}
          label={t(I18nKey.AUTOMATIONS$DETAIL$MODEL)}
        >
          {automation.model ?? t(I18nKey.COMMON$ACTIVE_PROFILE)}
        </ConfigField>

        {automation.notification && (
          <ConfigField
            icon={<BellIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$NOTIFICATION)}
          >
            {automation.notification}
          </ConfigField>
        )}
      </div>
    </SectionCard>
  );
}
