import { useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";
import CogIcon from "#/icons/cog.svg?react";
import GitBranchIcon from "#/icons/git-branch.svg?react";
import CheckCircleIcon from "#/icons/check-circle.svg?react";
import CalendarIcon from "#/icons/calendar.svg?react";
import SparkleIcon from "#/icons/sparkle.svg?react";
import UserIcon from "#/icons/user.svg?react";
import { Clock, Zap } from "lucide-react";
import BellIcon from "#/icons/bell.svg?react";
import CodeTagIcon from "#/icons/code-tag.svg?react";
import PuzzleIcon from "#/icons/puzzle.svg?react";
import LinkExternalIcon from "#/icons/link-external.svg?react";
import { formatEventOn } from "#/utils/automation-schedule";
import { automationRepositoryLabel } from "#/utils/automation-repo-sources";
import { SectionCard } from "./section-card";
import { ConfigField } from "./config-field";
import { BranchBadge } from "./branch-badge";
import { getAutomationPluginSpec } from "#/utils/automation-preset-sources";

interface ConfigurationSectionProps {
  automation: Automation;
  /**
   * Identity the automation runs as — the creator's email, or their raw user
   * id when the email could not be resolved. Omitted/null hides the field
   * (local backends, or while the lookup is still in flight).
   */
  runsAs?: string | null;
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

function ExpandableJsonValue({
  value,
  testId,
}: {
  value: unknown;
  testId: string;
}) {
  const { t } = useTranslation("openhands");
  const [expanded, setExpanded] = useState(false);
  const formatted = JSON.stringify(value, null, 2) ?? String(value);
  const isLong = formatted.length > 120;

  return (
    <div className="flex flex-col gap-1">
      <pre
        data-testid={testId}
        className="max-h-60 overflow-auto whitespace-pre-wrap break-all rounded-md bg-surface-raised p-2 font-mono text-xs text-content"
      >
        {isLong && !expanded ? formatted.slice(0, 120) + "…" : formatted}
      </pre>
      {isLong && (
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          aria-expanded={expanded}
          className="self-start text-xs text-muted hover:text-content"
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
  runsAs,
}: ConfigurationSectionProps) {
  const { t } = useTranslation("openhands");
  const isEvent = automation.trigger.type === "event";
  const pluginSpec = getAutomationPluginSpec(automation);

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
        {automation.repositories && automation.repositories.length > 0 && (
          <ConfigField
            icon={<GitBranchIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$REPOSITORIES)}
          >
            <ul className="flex flex-col gap-1">
              {automation.repositories.map((repository) => (
                <li
                  key={`${repository.url}@${repository.ref ?? ""}`}
                  data-testid="automation-repository"
                  className="flex flex-wrap items-center gap-1"
                >
                  <span className="break-all">
                    {automationRepositoryLabel(repository)}
                  </span>
                  {repository.ref && <BranchBadge branch={repository.ref} />}
                </li>
              ))}
            </ul>
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
            icon={<Zap className="size-3.5" aria-hidden="true" />}
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

        {pluginSpec && (
          <ConfigField
            icon={<PuzzleIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$PLUGIN_SPEC)}
          >
            <ExpandableJsonValue
              value={pluginSpec.plugin}
              testId="automation-plugin-spec"
            />
          </ConfigField>
        )}

        {pluginSpec && Object.keys(pluginSpec.config ?? {}).length > 0 && (
          <ConfigField
            icon={<CodeTagIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$PLUGIN_CONFIG)}
          >
            <ExpandableJsonValue
              value={pluginSpec.config}
              testId="automation-plugin-config"
            />
          </ConfigField>
        )}

        {automation.timeout != null && (
          <ConfigField
            icon={<Clock className="size-3.5" aria-hidden="true" />}
            label={t(I18nKey.SETTINGS$MCP_TIMEOUT_LABEL)}
          >
            <span data-testid="automation-timeout">{automation.timeout}</span>
          </ConfigField>
        )}

        {runsAs && (
          <ConfigField
            icon={<UserIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$RUNS_AS)}
          >
            <span className="break-all">{runsAs}</span>
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
