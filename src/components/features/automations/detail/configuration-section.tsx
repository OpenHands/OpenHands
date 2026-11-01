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
import { Zap } from "lucide-react";
import BellIcon from "#/icons/bell.svg?react";
import CodeTagIcon from "#/icons/code-tag.svg?react";
import LinkExternalIcon from "#/icons/link-external.svg?react";
import { formatEventOn } from "#/utils/automation-schedule";
import { automationRepositoryLabel } from "#/utils/automation-repo-sources";
import { SectionCard } from "./section-card";
import { ConfigField } from "./config-field";
import { BranchBadge } from "./branch-badge";

interface ConfigurationSectionProps {
  automation: Automation;
  /**
   * Identity the automation runs as — the creator's email, or their raw user
   * id when the email could not be resolved. Omitted/null hides the field
   * (local backends, or while the lookup is still in flight).
   */
  runsAs?: string | null;
}

const VALUE_TRUNCATE_LENGTH = 60;

function ExpandableValue({ value }: { value: string }) {
  const { t } = useTranslation("openhands");
  const [expanded, setExpanded] = useState(false);
  const isLong = value.length > VALUE_TRUNCATE_LENGTH;

  return (
    <div className="flex flex-col gap-1">
      <span className="font-mono break-all whitespace-pre-wrap">
        {isLong && !expanded
          ? `${value.slice(0, VALUE_TRUNCATE_LENGTH)}…`
          : value}
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
  runsAs,
}: ConfigurationSectionProps) {
  const { t } = useTranslation("openhands");
  const isEvent = automation.trigger.type === "event";
  const plugin = automation.plugin_spec?.plugin;
  const pluginConfig = Object.entries(automation.plugin_spec?.config ?? {});

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
            <ExpandableValue value={automation.trigger.filter} />
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

        {automation.timeout != null && (
          <ConfigField
            icon={<CalendarIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATION_SETUP$TIMEOUT_SECONDS)}
          >
            {automation.timeout}
          </ConfigField>
        )}

        {plugin && (
          <>
            <ConfigField
              icon={<CodeTagIcon className="size-3.5" />}
              label={t(I18nKey.AUTOMATION_SETUP$PLUGIN_SOURCE)}
            >
              <span className="break-all">{plugin.source}</span>
            </ConfigField>
            {plugin.ref && (
              <ConfigField
                icon={<GitBranchIcon className="size-3.5" />}
                label={t(I18nKey.AUTOMATION_SETUP$PLUGIN_REF)}
              >
                <span className="break-all">{plugin.ref}</span>
              </ConfigField>
            )}
            {plugin.repo_path && (
              <ConfigField
                icon={<CodeTagIcon className="size-3.5" />}
                label={t(I18nKey.SETTINGS$PLUGINS_REPO_PATH_LABEL)}
              >
                <span className="break-all">{plugin.repo_path}</span>
              </ConfigField>
            )}
          </>
        )}

        {pluginConfig.length > 0 && (
          <div className="col-span-2">
            <ConfigField
              icon={<CodeTagIcon className="size-3.5" />}
              label={t(I18nKey.AUTOMATIONS$DETAIL$PLUGIN_CONFIG)}
            >
              <dl
                data-testid="automation-plugin-config"
                className="flex flex-col gap-3"
              >
                {pluginConfig.map(([key, value]) => (
                  <div key={key}>
                    <dt className="text-xs text-muted break-all">{key}</dt>
                    <dd>
                      <ExpandableValue
                        value={
                          typeof value === "string"
                            ? value
                            : (JSON.stringify(value, null, 2) ?? String(value))
                        }
                      />
                    </dd>
                  </div>
                ))}
              </dl>
            </ConfigField>
          </div>
        )}

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
