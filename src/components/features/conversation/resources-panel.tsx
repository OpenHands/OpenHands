import { useSyncExternalStore, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import ConversationResourcesService, {
  type ConversationResources,
} from "#/api/conversation-resources-service";
import {
  agentProfileSupportsSecretRefs,
  agentProfileSupportsSkills,
} from "#/api/agent-profiles-service/profile-field-support";
import {
  getMcpHealthSnapshot,
  subscribeMcpHealth,
} from "#/api/mcp-health/mcp-health-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { CONVERSATION_RESOURCES_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";
import { NavigationLink } from "#/components/shared/navigation-link";
import { groupSkillsByScope, SKILL_SCOPE_ORDER } from "#/utils/skill-scope";
import type { McpServerHealth } from "#/types/mcp-health";

function ResourceRow({
  title,
  settings,
  children,
}: {
  title: string;
  settings: string;
  children: ReactNode;
}) {
  const { t } = useTranslation("openhands");
  return (
    <details className="rounded-md border border-border bg-surface-raised">
      <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-contrast">
        {title}
      </summary>
      <div className="flex flex-col gap-2 px-3 pb-3 text-sm break-words">
        {children}
        <NavigationLink
          to={settings}
          className="self-start text-xs text-muted underline hover:text-contrast"
        >
          {t(I18nKey.SETTINGS$TITLE)}
        </NavigationLink>
      </div>
    </details>
  );
}

function McpResource({
  server,
}: {
  server: NonNullable<ConversationResources["mcpServers"]>[number];
}) {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const health: McpServerHealth | undefined = useSyncExternalStore(
    subscribeMcpHealth,
    () => getMcpHealthSnapshot()[server.healthKey],
  );
  // A Cloud stdio probe cannot validate the sandbox process from the app server.
  const status =
    backend.kind === "cloud" && server.type === "stdio" ? undefined : health;
  let label = t(I18nKey.MCP$HEALTH_STATUS_UNCHECKED);
  if (status?.status === "checking")
    label = t(I18nKey.MCP$HEALTH_STATUS_CHECKING);
  if (status?.status === "healthy")
    label =
      status.verification === "verified"
        ? t(I18nKey.MCP$HEALTH_STATUS_VERIFIED, { count: status.toolCount })
        : t(I18nKey.MCP$HEALTH_STATUS_CONNECTIVITY_ONLY);
  if (status?.status === "failed") label = t(I18nKey.RESOURCES$CHECK_FAILED);
  return (
    <li className="flex flex-col gap-1">
      <span>{server.name}</span>
      <span className="text-xs text-muted">{label}</span>
    </li>
  );
}

const SKILL_GROUP_LABELS = {
  project: I18nKey.SKILLS_MODAL$SECTION_PROJECT,
  personal: I18nKey.SETTINGS$SKILLS_SOURCE_PERSONAL,
  public: I18nKey.SKILLS_MODAL$SECTION_PUBLIC,
};

export function ResourcesPanel() {
  const { t } = useTranslation("openhands");
  const active = useActiveBackend();
  const conversation = useActiveConversation();
  const current = conversation.data;
  // Scope identity is in the key; backend/session credentials must stay out of it.
  // eslint-disable-next-line @tanstack/query/exhaustive-deps
  const resources = useQuery({
    queryKey: CONVERSATION_RESOURCES_QUERY_KEYS.detail(
      active.backend.id,
      active.backend.connectionRevision ?? 0,
      active.orgId,
      current?.id,
      current?.conversation_url,
    ),
    queryFn: () =>
      ConversationResourcesService.load(current!.id, active, {
        conversationUrl: current!.conversation_url,
        sessionApiKey: current!.session_api_key,
      }),
    enabled: !!current,
    staleTime: 0,
    refetchInterval: 30_000,
    retry: false,
    meta: { disableToast: true },
  });
  const unavailable = (
    <p className="text-muted">{t(I18nKey.RESOURCES$UNAVAILABLE)}</p>
  );
  const none = (
    <p className="text-muted">{t(I18nKey.CONVERSATION$OVERVIEW_NONE)}</p>
  );
  const data = resources.data;

  if (
    conversation.isError ||
    resources.isError ||
    (!conversation.isPending && !current)
  ) {
    return (
      <div role="status" className="flex flex-col gap-3 p-3 text-sm">
        <p>{t(I18nKey.RESOURCES$LOAD_ERROR)}</p>
        <button
          type="button"
          className="self-start underline"
          onClick={() => {
            if (current && !conversation.isError) void resources.refetch();
            else void conversation.refetch();
          }}
        >
          {t(I18nKey.SETTINGS$APPS_VIEW_RETRY)}
        </button>
      </div>
    );
  }
  if (!data)
    return (
      <p role="status" className="p-3 text-sm text-muted">
        {t(I18nKey.HOME$LOADING)}
      </p>
    );

  const changed =
    data.profile?.currentRevision != null &&
    data.profile.currentRevision !== data.profile.launchedRevision;
  const isAcp = data.agentKind === "acp";
  const skillGroups = groupSkillsByScope(data.skills ?? []);
  return (
    <main
      data-testid="resources-panel"
      className="h-full overflow-y-auto custom-scrollbar-always flex flex-col gap-3 p-3 text-sm text-contrast"
    >
      <p className="text-xs text-muted">{t(I18nKey.RESOURCES$SNAPSHOT_HINT)}</p>
      <ResourceRow
        title={t(I18nKey.RESOURCES$PROFILE)}
        settings="/settings/agents"
      >
        {data.profile ? (
          <>
            <p>{data.profile.name ?? data.profile.id}</p>
            <p>
              {t(I18nKey.RESOURCES$REVISION, {
                revision: data.profile.launchedRevision,
              })}
            </p>
            {changed && (
              <p role="status" className="text-feedback-warning">
                {t(I18nKey.RESOURCES$PROFILE_CHANGED, {
                  revision: data.profile.currentRevision,
                })}
              </p>
            )}
            {data.profile.currentRevision == null && unavailable}
          </>
        ) : (
          <p>{t(I18nKey.RESOURCES$NO_PROFILE)}</p>
        )}
      </ResourceRow>
      <ResourceRow
        title={t(I18nKey.SETTINGS$NAV_LLM)}
        settings={isAcp ? "/settings/agents" : "/settings/llm"}
      >
        {data.model ? <p>{data.model}</p> : unavailable}
      </ResourceRow>
      <ResourceRow title={t(I18nKey.SETTINGS$SKILLS_TITLE)} settings="/skills">
        {isAcp || !agentProfileSupportsSkills()
          ? unavailable
          : data.skills === null
            ? unavailable
            : data.skills.length === 0
              ? none
              : SKILL_SCOPE_ORDER.map(
                  (scope) =>
                    skillGroups[scope].length > 0 && (
                      <div key={scope}>
                        <p className="text-xs text-muted">
                          {t(SKILL_GROUP_LABELS[scope])}
                        </p>
                        <ul>
                          {skillGroups[scope].map((skill) => (
                            <li key={skill.name}>{skill.name}</li>
                          ))}
                        </ul>
                      </div>
                    ),
                )}
      </ResourceRow>
      <ResourceRow title={t(I18nKey.SETTINGS$MCP_TITLE)} settings="/mcp">
        {data.mcpServers === null ? (
          unavailable
        ) : data.mcpServers.length === 0 ? (
          none
        ) : (
          <>
            <ul className="flex flex-col gap-2">
              {data.mcpServers.map((server) => (
                <McpResource key={server.name} server={server} />
              ))}
            </ul>
            <p className="text-xs text-muted">
              {t(I18nKey.RESOURCES$HEALTH_HINT)}
            </p>
          </>
        )}
      </ResourceRow>
      <ResourceRow
        title={t(I18nKey.SETTINGS$NAV_SECRETS)}
        settings="/settings/secrets"
      >
        {!agentProfileSupportsSecretRefs() || data.secretNames === null ? (
          unavailable
        ) : data.secretNames.length === 0 ? (
          none
        ) : (
          <ul>
            {data.secretNames.map((name) => (
              <li key={name}>{name}</li>
            ))}
          </ul>
        )}
      </ResourceRow>
      <ResourceRow
        title={t(I18nKey.SETTINGS$NAV_CONDENSER)}
        settings={
          active.backend.kind === "cloud"
            ? "/settings/condenser"
            : "/settings/agents"
        }
      >
        {isAcp || !data.condenser ? (
          unavailable
        ) : (
          <>
            <p>{data.condenser.kind}</p>
            <dl className="grid grid-cols-2 gap-2">
              {data.condenser.maxSize != null && (
                <>
                  <dt>{t(I18nKey.SCHEMA$CONDENSER$MAX_SIZE$LABEL)}</dt>
                  <dd>{data.condenser.maxSize}</dd>
                </>
              )}
              {data.condenser.keepFirst != null && (
                <>
                  <dt>{t(I18nKey.SCHEMA$CONDENSER$KEEP_FIRST$LABEL)}</dt>
                  <dd>{data.condenser.keepFirst}</dd>
                </>
              )}
              {data.condenser.maxTokens != null && (
                <>
                  <dt>{t(I18nKey.SCHEMA$CONDENSER$MAX_TOKENS$LABEL)}</dt>
                  <dd>{data.condenser.maxTokens}</dd>
                </>
              )}
            </dl>
          </>
        )}
      </ResourceRow>
      <ResourceRow
        title={t(I18nKey.SETTINGS$NAV_VERIFICATION)}
        settings="/settings/verification"
      >
        {isAcp || !data.verification ? (
          unavailable
        ) : data.verification.kind ? (
          <>
            <p>{data.verification.kind}</p>
            <dl className="grid grid-cols-2 gap-2">
              {data.verification.mode && (
                <>
                  <dt>{t(I18nKey.SCHEMA$VERIFICATION$CRITIC_MODE$LABEL)}</dt>
                  <dd>{data.verification.mode}</dd>
                </>
              )}
              {data.verification.model && (
                <>
                  <dt>
                    {t(I18nKey.SCHEMA$VERIFICATION$CRITIC_MODEL_NAME$LABEL)}
                  </dt>
                  <dd>{data.verification.model}</dd>
                </>
              )}
              {data.verification.threshold != null && (
                <>
                  <dt>
                    {t(I18nKey.SCHEMA$VERIFICATION$CRITIC_THRESHOLD$LABEL)}
                  </dt>
                  <dd>{data.verification.threshold}</dd>
                </>
              )}
              {data.verification.maxIterations != null && (
                <>
                  <dt>
                    {t(
                      I18nKey.SCHEMA$VERIFICATION$MAX_REFINEMENT_ITERATIONS$LABEL,
                    )}
                  </dt>
                  <dd>{data.verification.maxIterations}</dd>
                </>
              )}
            </dl>
          </>
        ) : (
          <p>{t(I18nKey.SETTINGS$SKILLS_DISABLED)}</p>
        )}
      </ResourceRow>
    </main>
  );
}
