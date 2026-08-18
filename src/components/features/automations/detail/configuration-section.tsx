import { useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { Automation, AutomationRepoSource } from "#/types/automation";
import CogIcon from "#/icons/cog.svg?react";
import GitBranchIcon from "#/icons/git-branch.svg?react";
import CheckCircleIcon from "#/icons/check-circle.svg?react";
import CalendarIcon from "#/icons/calendar.svg?react";
import SparkleIcon from "#/icons/sparkle.svg?react";
import { Zap } from "lucide-react";
import CodeTagIcon from "#/icons/code-tag.svg?react";
import LinkExternalIcon from "#/icons/link-external.svg?react";
import ClockIcon from "#/icons/clock.svg?react";
import PowerIcon from "#/icons/power.svg?react";
import { formatEventOn } from "#/utils/automation-schedule";
import { useDeploymentCapabilities } from "#/hooks/query/use-manifest-capabilities";
import { SectionCard } from "./section-card";
import { ConfigField } from "./config-field";
import { BranchBadge } from "./branch-badge";
import { AdvancedDisclosure } from "./advanced-disclosure";
import { WebhooksLinkCard } from "./webhooks-link-card";

interface ConfigurationSectionProps {
  automation: Automation;
}

const FILTER_TRUNCATE_LENGTH = 60;

/** Short display label for a repo URL: "owner/repo" when it can be parsed, the raw URL otherwise. */
function repoLabel(url: string): string {
  const match = url.match(/[/:]([\w.-]+\/[\w.-]+?)(?:\.git)?$/);
  return match ? match[1] : url;
}

function RepoRow({ repo }: { repo: AutomationRepoSource }) {
  return (
    <span className="flex items-center gap-1">
      {repoLabel(repo.url)}
      {repo.ref && <BranchBadge branch={repo.ref} />}
    </span>
  );
}

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
  const { data: capabilities } = useDeploymentCapabilities();
  const features = capabilities?.features ?? [];

  let scheduleDisplay = automation.trigger.schedule ?? "";
  if (automation.trigger.schedule_human) {
    scheduleDisplay = automation.timezone
      ? `${automation.trigger.schedule_human} (${automation.timezone})`
      : automation.trigger.schedule_human;
  }

  const triggerDisplay = isEvent
    ? t(I18nKey.AUTOMATIONS$DETAIL$TRIGGER_EVENT)
    : t(I18nKey.AUTOMATIONS$DETAIL$TRIGGER_SCHEDULE);

  // Prefer the full repo list recorded at creation time; fall back to the
  // single legacy repository/branch fields for older automations.
  const repos = automation.preset_metadata?.repos;
  const hasRepos = (repos && repos.length > 0) || automation.repository;

  return (
    <SectionCard
      icon={<CogIcon className="size-4" />}
      title={t(I18nKey.AUTOMATIONS$DETAIL$CONFIGURATION)}
    >
      <div className="grid grid-cols-2 gap-x-4 gap-y-5">
        {hasRepos && (
          <ConfigField
            icon={<GitBranchIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$REPOSITORIES)}
          >
            {repos && repos.length > 0 ? (
              <div className="flex flex-col gap-1.5">
                {repos.map((repo) => (
                  <RepoRow key={repo.url} repo={repo} />
                ))}
              </div>
            ) : (
              <span className="flex items-center gap-1">
                {automation.repository}
                {automation.branch && (
                  <BranchBadge branch={automation.branch} />
                )}
              </span>
            )}
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

        <ConfigField
          icon={<SparkleIcon className="size-3.5" />}
          label={t(I18nKey.AUTOMATIONS$DETAIL$MODEL)}
        >
          {automation.model ?? t(I18nKey.COMMON$ACTIVE_PROFILE)}
        </ConfigField>

        <AdvancedDisclosure testId="configuration-advanced">
          <ConfigField
            icon={<ClockIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$TIMEOUT)}
          >
            {automation.timeout != null
              ? t(I18nKey.AUTOMATIONS$DETAIL$TIMEOUT_SECONDS, {
                  seconds: automation.timeout,
                })
              : t(I18nKey.AUTOMATIONS$DETAIL$TIMEOUT_DEFAULT)}
          </ConfigField>

          <ConfigField
            icon={<PowerIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$KEEP_ALIVE)}
          >
            {automation.keep_alive
              ? t(I18nKey.AUTOMATIONS$DETAIL$KEEP_ALIVE_ON)
              : t(I18nKey.AUTOMATIONS$DETAIL$KEEP_ALIVE_OFF)}
          </ConfigField>

          {features.includes("webhookDelivery") && <WebhooksLinkCard />}
        </AdvancedDisclosure>
      </div>
    </SectionCard>
  );
}
