import { Fragment, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router";
import { useTranslation } from "react-i18next";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import {
  agentProfileSupportsSecretRefs,
  agentProfileSupportsSkills,
} from "#/api/agent-profiles-service/profile-field-support";
import { SecretsService } from "#/api/secrets-service";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useAgentProfiles } from "#/hooks/query/use-agent-profiles";
import {
  AGENT_PROFILES_QUERY_KEYS,
  CONFIG_CACHE_OPTIONS,
  AGENT_PROFILES_RETRY_OPTIONS,
} from "#/hooks/query/query-keys";
import { useConversationSkills } from "#/hooks/query/use-conversation-skills";
import { useSettings } from "#/hooks/query/use-settings";
import { useMcpServerHealth } from "#/hooks/use-mcp-server-health";
import { I18nKey } from "#/i18n/declaration";
import type { MCPServerConfig } from "#/types/mcp-server";
import { flattenMcpConfig } from "#/utils/mcp-installed-servers";
import { groupSkillsByScope, SKILL_SCOPE_ORDER } from "#/utils/skill-scope";
import {
  buildSkillEnablementFilter,
  profileSkillEnablement,
} from "#/utils/skill-enablement";

function ResourceSection({
  title,
  to,
  children,
}: {
  title: string;
  to: string;
  children: ReactNode;
}) {
  const { t } = useTranslation("openhands");
  return (
    <details className="border-b border-border py-3">
      <summary className="cursor-pointer text-sm font-medium">{title}</summary>
      <div className="mt-3 space-y-2 text-sm break-words">
        {children}
        <Link to={to} className="block text-muted underline">
          {t(I18nKey.SETTINGS$TITLE)}
        </Link>
      </div>
    </details>
  );
}

function ServerResource({ server }: { server: MCPServerConfig }) {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const { health } = useMcpServerHealth(server);
  const status =
    backend.kind === "cloud" && server.type === "stdio"
      ? t(I18nKey.RESOURCES$UNAVAILABLE)
      : health.status === "healthy"
        ? t(
            health.verification === "verified"
              ? I18nKey.MCP$HEALTH_STATUS_VERIFIED
              : I18nKey.MCP$HEALTH_STATUS_CONNECTIVITY_ONLY,
            { count: health.toolCount },
          )
        : t(
            health.status === "failed"
              ? I18nKey.ERROR$GENERIC
              : health.status === "checking"
                ? I18nKey.MCP$HEALTH_STATUS_CHECKING
                : I18nKey.MCP$HEALTH_STATUS_UNCHECKED,
          );
  return (
    <li>
      {server.name ?? server.id}
      <span className="block text-muted">{status}</span>
    </li>
  );
}

export function ConversationResourcesPanel() {
  const { t } = useTranslation("openhands");
  const { backend, orgId } = useActiveBackend();
  const conversationQuery = useActiveConversation();
  const conversation = conversationQuery.data;
  const launched = conversation?.launched_agent_profile;
  const profiles = useAgentProfiles({ enabled: !!launched });
  const summary = profiles.data?.profiles.find(
    (entry) => entry.id === launched?.agent_profile_id,
  );
  const detail = useQuery({
    queryKey: AGENT_PROFILES_QUERY_KEYS.detail(
      backend.id,
      orgId,
      summary?.name ?? "",
    ),
    queryFn: () => AgentProfilesService.getProfile(summary!.name),
    enabled: !!summary,
    ...CONFIG_CACHE_OPTIONS,
    ...AGENT_PROFILES_RETRY_OPTIONS,
    meta: { disableToast: true },
  });
  const profile = detail.data?.profile;
  const changed = !!profile && profile.revision !== launched?.revision;
  const applicableProfile = !changed ? profile : undefined;
  const openHandsProfile =
    conversation?.agent_kind !== "acp" &&
    applicableProfile?.agent_kind === "openhands"
      ? applicableProfile
      : undefined;
  const {
    data: settings,
    isError: settingsError,
    isLoading: settingsLoading,
  } = useSettings();
  const skills = useConversationSkills();
  const secrets = useQuery({
    queryKey: ["conversation-resource-secrets", backend.id, orgId],
    queryFn: SecretsService.getSecretsOrThrow,
    enabled:
      !!launched &&
      agentProfileSupportsSecretRefs() &&
      launched.secret_refs == null,
    ...CONFIG_CACHE_OPTIONS,
    retry: false,
    meta: { disableToast: true },
  });
  const unavailable = (
    <p className="text-muted">{t(I18nKey.RESOURCES$UNAVAILABLE)}</p>
  );
  const pending = <p role="status">{t(I18nKey.HOME$LOADING)}</p>;
  const unsupported = (
    <p className="text-muted">{t(I18nKey.RESOURCES$NOT_APPLICABLE)}</p>
  );
  const empty = (
    <p className="text-muted">{t(I18nKey.CONVERSATION$OVERVIEW_NONE)}</p>
  );
  const isAcp = conversation?.agent_kind === "acp";
  const canReadSkills =
    !isAcp && (!agentProfileSupportsSkills() || !!openHandsProfile);
  const enabledSkill = openHandsProfile
    ? buildSkillEnablementFilter(profileSkillEnablement(openHandsProfile))
    : () => true;
  const grouped = groupSkillsByScope(
    (skills.data ?? []).filter((skill) => enabledSkill(skill.name)),
    conversation?.selected_workspace,
  );
  const scopeLabels = {
    project: I18nKey.SKILLS_MODAL$SECTION_PROJECT,
    personal: I18nKey.SKILLS_MODAL$SECTION_USER,
    public: I18nKey.SKILLS_MODAL$SECTION_PUBLIC,
  };
  const servers = flattenMcpConfig(settings?.mcp_config ?? {}).filter(
    (server) =>
      server.enabled !== false &&
      (applicableProfile?.mcp_server_refs == null ||
        applicableProfile.mcp_server_refs.includes(server.id)),
  );
  const secretNames =
    launched?.secret_refs ?? secrets.data?.map((secret) => secret.name);
  const loading =
    conversationQuery.isLoading ||
    (!!launched && (profiles.isLoading || detail.isLoading));

  const condenser = openHandsProfile?.condenser;
  const condenserKind =
    condenser &&
    typeof condenser === "object" &&
    "condenser_kind" in condenser &&
    typeof condenser.condenser_kind === "string"
      ? condenser.condenser_kind
      : null;

  const condenserFields = [
    ["max_size", I18nKey.SCHEMA$CONDENSER$MAX_SIZE$LABEL],
    ["max_tokens", I18nKey.SCHEMA$CONDENSER$MAX_TOKENS$LABEL],
    ["keep_first", I18nKey.SCHEMA$CONDENSER$KEEP_FIRST$LABEL],
    ["minimum_progress", I18nKey.SCHEMA$CONDENSER$MINIMUM_PROGRESS$LABEL],
    [
      "hard_context_reset_max_retries",
      I18nKey.SCHEMA$CONDENSER$HARD_CONTEXT_RESET_MAX_RETRIES$LABEL,
    ],
    [
      "hard_context_reset_context_scaling",
      I18nKey.SCHEMA$CONDENSER$HARD_CONTEXT_RESET_CONTEXT_SCALING$LABEL,
    ],
  ] as const;

  return (
    <div className="h-full overflow-auto p-4 custom-scrollbar">
      <h2 className="text-base font-medium">{t(I18nKey.RESOURCES$TITLE)}</h2>
      {loading ? <p role="status">{t(I18nKey.HOME$LOADING)}</p> : null}
      <ResourceSection
        title={t(I18nKey.CONVERSATION_PANEL$PREVIEW_AGENT_PROFILE)}
        to="/settings/agents"
      >
        {launched ? (
          <>
            <p>{summary?.name ?? launched.agent_profile_id}</p>
            <p>
              {t(I18nKey.RESOURCES$REVISION, { revision: launched.revision })}
            </p>
            {changed ? (
              <p role="status">{t(I18nKey.RESOURCES$CHANGED)}</p>
            ) : null}
          </>
        ) : (
          <p>{t(I18nKey.RESOURCES$NO_PROFILE)}</p>
        )}
        {profiles.isError || detail.isError ? unavailable : null}
      </ResourceSection>
      <ResourceSection
        title={t(I18nKey.SETTINGS$AGENT_MODEL)}
        to="/settings/llm"
      >
        {conversation?.llm_model ? (
          <p>{conversation.llm_model}</p>
        ) : (
          unavailable
        )}
      </ResourceSection>
      <ResourceSection title={t(I18nKey.SETTINGS$SKILLS_TITLE)} to="/skills">
        {!canReadSkills
          ? isAcp
            ? unsupported
            : unavailable
          : skills.isError
            ? unavailable
            : SKILL_SCOPE_ORDER.map((scope) =>
                grouped[scope].length > 0 ? (
                  <div key={scope}>
                    <h3 className="text-muted">{t(scopeLabels[scope])}</h3>
                    <ul>
                      {grouped[scope].map((skill) => (
                        <li key={skill.name}>{skill.name}</li>
                      ))}
                    </ul>
                  </div>
                ) : null,
              )}
        {canReadSkills &&
        skills.isSuccess &&
        Object.values(grouped).every((items) => items.length === 0)
          ? empty
          : null}
      </ResourceSection>
      <ResourceSection title={t(I18nKey.SETTINGS$AGENT_PROFILE_MCP)} to="/mcp">
        {(!applicableProfile && !isAcp) || settingsError ? (
          unavailable
        ) : settingsLoading ? (
          pending
        ) : servers.length ? (
          <ul className="space-y-2">
            {servers.map((server) => (
              <ServerResource key={server.id} server={server} />
            ))}
          </ul>
        ) : (
          empty
        )}
      </ResourceSection>
      <ResourceSection
        title={t(I18nKey.SETTINGS$NAV_SECRETS)}
        to="/settings/secrets"
      >
        {!agentProfileSupportsSecretRefs() ? (
          unsupported
        ) : launched && secrets.isLoading && launched.secret_refs == null ? (
          pending
        ) : !launched || secrets.isError || !secretNames ? (
          unavailable
        ) : secretNames.length ? (
          <ul>
            {secretNames.map((name) => (
              <li key={name}>{name}</li>
            ))}
          </ul>
        ) : (
          empty
        )}
      </ResourceSection>
      <ResourceSection
        title={t(I18nKey.SETTINGS$NAV_CONDENSER)}
        to={
          backend.kind === "cloud" ? "/settings/condenser" : "/settings/agents"
        }
      >
        {isAcp ? (
          unsupported
        ) : openHandsProfile ? (
          <dl>
            {condenser &&
            typeof condenser === "object" &&
            "enabled" in condenser &&
            typeof condenser.enabled === "boolean" ? (
              <>
                <dt>{t(I18nKey.SCHEMA$CONDENSER$ENABLED$LABEL)}</dt>
                <dd>
                  {t(
                    condenser.enabled
                      ? I18nKey.SETTINGS$SKILLS_ENABLED
                      : I18nKey.SETTINGS$SKILLS_DISABLED,
                  )}
                </dd>
              </>
            ) : null}
            <dt>{t(I18nKey.SCHEMA$CONDENSER$CONDENSER_KIND$LABEL)}</dt>
            <dd>{condenserKind ?? t(I18nKey.CONVERSATION$OVERVIEW_NONE)}</dd>
            {condenserFields.map(([field, label]) => {
              const value =
                condenser && typeof condenser === "object" && field in condenser
                  ? (condenser as Record<string, unknown>)[field]
                  : undefined;
              return typeof value === "number" ? (
                <Fragment key={field}>
                  <dt>{t(label)}</dt>
                  <dd>{value}</dd>
                </Fragment>
              ) : null;
            })}
          </dl>
        ) : (
          unavailable
        )}
      </ResourceSection>
      <ResourceSection
        title={t(I18nKey.SETTINGS$NAV_VERIFICATION)}
        to="/settings/verification"
      >
        {isAcp ? (
          unsupported
        ) : openHandsProfile ? (
          <dl>
            <dt>{t(I18nKey.SCHEMA$VERIFICATION$CRITIC_MODE$LABEL)}</dt>
            <dd>{openHandsProfile.verification.critic_mode}</dd>
            <dt>{t(I18nKey.SCHEMA$VERIFICATION$CRITIC_THRESHOLD$LABEL)}</dt>
            <dd>{openHandsProfile.verification.critic_threshold}</dd>
            <dt>
              {t(I18nKey.SCHEMA$VERIFICATION$MAX_REFINEMENT_ITERATIONS$LABEL)}
            </dt>
            <dd>{openHandsProfile.verification.max_refinement_iterations}</dd>
            <dt>{t(I18nKey.SCHEMA$VERIFICATION$CRITIC_MODEL_NAME$LABEL)}</dt>
            <dd>
              {openHandsProfile.verification.critic_model_name ??
                t(I18nKey.CONVERSATION$OVERVIEW_NONE)}
            </dd>
            <dt>{t(I18nKey.SCHEMA$VERIFICATION$CRITIC_ENABLED$LABEL)}</dt>
            <dd>
              {t(
                openHandsProfile.verification.critic_enabled
                  ? I18nKey.SETTINGS$SKILLS_ENABLED
                  : I18nKey.SETTINGS$SKILLS_DISABLED,
              )}
            </dd>
            <dt>
              {t(I18nKey.SCHEMA$VERIFICATION$ENABLE_ITERATIVE_REFINEMENT$LABEL)}
            </dt>
            <dd>
              {t(
                openHandsProfile.verification.enable_iterative_refinement
                  ? I18nKey.SETTINGS$SKILLS_ENABLED
                  : I18nKey.SETTINGS$SKILLS_DISABLED,
              )}
            </dd>
          </dl>
        ) : (
          unavailable
        )}
      </ResourceSection>
    </div>
  );
}
