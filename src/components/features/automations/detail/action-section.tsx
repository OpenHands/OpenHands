import { useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { Automation } from "#/types/automation";
import TerminalIcon from "#/icons/terminal.svg?react";
import GitBranchIcon from "#/icons/git-branch.svg?react";
import SparkleIcon from "#/icons/sparkle.svg?react";
import CodeTagIcon from "#/icons/code-tag.svg?react";
import PuzzleIcon from "#/icons/puzzle.svg?react";
import UserIcon from "#/icons/user.svg?react";
import { useAgentProfiles } from "#/hooks/query/use-agent-profiles";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { automationRepositoryLabel } from "#/utils/automation-repo-sources";
import { SectionCard } from "./section-card";
import { ConfigField } from "./config-field";
import { BranchBadge } from "./branch-badge";
import { PluginChip } from "./plugin-chip";
import { PromptBody } from "./prompt-section";
import { ScriptBody } from "./script-section";

/**
 * An automation run from an agent profile: the profile, then the LLM profile
 * it brings, shown the way the setup form shows them.
 */
function AgentProfileFields({ id }: { id: string }) {
  const { t } = useTranslation("openhands");
  const { data, isLoading } = useAgentProfiles();
  const { data: llmProfiles } = useLlmProfiles();
  const profile = data?.profiles?.find((candidate) => candidate.id === id);
  let name: string;
  if (profile) name = profile.name;
  else if (isLoading) name = "…";
  else name = t(I18nKey.AUTOMATIONS$UNAVAILABLE_AGENT_PROFILE, { id });
  // An ACP agent runs its own model; an agent profile without its own LLM
  // profile runs on the active one.
  const llmProfile =
    profile?.agent_kind === "acp"
      ? t(I18nKey.CONVERSATION$ACP_AGENT_GENERIC)
      : profile?.llm_profile_ref ||
        llmProfiles?.active_profile ||
        t(I18nKey.COMMON$ACTIVE_PROFILE);

  return (
    <>
      <ConfigField
        icon={<UserIcon className="size-3.5" />}
        label={t(I18nKey.CHAT$AGENT_PROFILE_PLACEHOLDER)}
      >
        <span data-testid="automation-agent-profile-name">{name}</span>
      </ConfigField>
      <ConfigField
        icon={<SparkleIcon className="size-3.5" />}
        label={t(I18nKey.AUTOMATIONS$DETAIL$MODEL)}
      >
        <span data-testid="automation-agent-profile-llm">{llmProfile}</span>
        <span className="mt-0.5 block text-xs text-muted">
          {t(I18nKey.AUTOMATIONS$DETAIL$FROM_AGENT_PROFILE)}
        </span>
      </ConfigField>
    </>
  );
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

/**
 * What the automation does once triggered: its prompt or script, then the
 * model, repositories and plugins it runs with. Mirrors the setup form's
 * action group.
 */
export function ActionSection({ automation }: { automation: Automation }) {
  const { t } = useTranslation("openhands");
  const repositories = automation.repositories ?? [];
  const plugins = automation.plugins ?? [];
  const plugin = automation.plugin_spec?.plugin;
  const pluginConfig = Object.entries(automation.plugin_spec?.config ?? {});

  return (
    <SectionCard
      testId="automation-action-section"
      icon={<TerminalIcon className="size-4" />}
      title={t(I18nKey.AUTOMATIONS$SECTION$ACTION)}
      meta={
        automation.prompt
          ? t(I18nKey.AUTOMATIONS$DETAIL$PROMPT)
          : t(I18nKey.AUTOMATIONS$DETAIL$SCRIPT)
      }
    >
      {automation.prompt ? (
        <PromptBody prompt={automation.prompt} />
      ) : (
        <ScriptBody automation={automation} />
      )}

      <div className="mt-5 grid grid-cols-2 gap-x-4 gap-y-5 border-t border-border pt-5">
        {/* One or the other, as in the setup form: an agent profile brings
            its own LLM profile. */}
        {automation.agent_profile_id ? (
          <AgentProfileFields id={automation.agent_profile_id} />
        ) : (
          <ConfigField
            icon={<SparkleIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$MODEL)}
          >
            {automation.model ?? t(I18nKey.COMMON$ACTIVE_PROFILE)}
          </ConfigField>
        )}

        {repositories.length > 0 ? (
          <ConfigField
            icon={<GitBranchIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$REPOSITORIES)}
          >
            <ul className="flex flex-col gap-1">
              {repositories.map((repository) => (
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
        ) : null}

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

        {plugins.length > 0 ? (
          <ConfigField
            icon={<PuzzleIcon className="size-3.5" />}
            label={t(I18nKey.AUTOMATIONS$DETAIL$PLUGINS)}
          >
            <div className="flex flex-wrap gap-2">
              {plugins.map((plugin) => (
                <PluginChip key={plugin} name={plugin} />
              ))}
            </div>
          </ConfigField>
        ) : null}
      </div>
    </SectionCard>
  );
}
