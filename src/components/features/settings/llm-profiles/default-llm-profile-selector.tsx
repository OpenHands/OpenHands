import { useMemo } from "react";
import { Info } from "lucide-react";
import { useTranslation } from "react-i18next";
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME } from "#/api/agent-profiles-service/agent-profiles-service.api";
import type { ProfileInfo } from "#/api/profiles-service/profiles-service.api";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
import { useAgentProfiles } from "#/hooks/query/use-agent-profiles";
import { useFreeModels } from "#/hooks/query/use-free-models";
import { formatModelNameForDisplay } from "#/utils/format-model-name";
import { settingsListContainerClassName } from "#/utils/settings-list-classes";
import { cn } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";

interface DefaultLlmProfileSelectorProps {
  profiles: ProfileInfo[];
  /** The account's active (default) LLM profile name. */
  activeProfile: string | null;
  /** Used to name each profile's provider connection in the options. */
  connections: ProviderConnection[];
  /** When false, the selector is read-only (cloud view-only members). */
  canManage: boolean;
  isActivating: boolean;
  onSelect: (name: string) => void;
}

/**
 * Always-visible control for the default LLM profile — the account's active
 * LLM profile, which the `default` agent profile launches new conversations
 * with. Each option names the profile, its provider connection and its model.
 * When the active agent profile launches with a different LLM, a note says so,
 * mirroring `resolveConversationLaunch`.
 */
export function DefaultLlmProfileSelector({
  profiles,
  activeProfile,
  connections,
  canManage,
  isActivating,
  onSelect,
}: DefaultLlmProfileSelectorProps) {
  const { t } = useTranslation("openhands");
  const freeModels = useFreeModels();
  const { data: agentProfiles } = useAgentProfiles();

  const items = useMemo(() => {
    const connectionNames = new Map(
      connections.map((connection) => [connection.id, connection.display_name]),
    );
    return profiles.map((profile) => {
      const connectionName = profile.provider_connection_id
        ? connectionNames.get(profile.provider_connection_id)
        : undefined;
      // Without a connection, a `provider/model` id still names its provider.
      const modelProvider = profile.model?.includes("/")
        ? profile.model.split("/")[0]
        : undefined;
      const provider =
        connectionName ??
        modelProvider ??
        t(I18nKey.SETTINGS$PROFILE_UNLINKED_PROVIDER);
      const model = formatModelNameForDisplay(profile.model, freeModels);
      return {
        key: profile.name,
        label: profile.name,
        description: model ? `${provider} · ${model}` : provider,
      };
    });
  }, [profiles, connections, freeModels, t]);

  const selected = items.find((item) => item.key === activeProfile);

  const overrideNote = useMemo(() => {
    const activeAgent = agentProfiles?.profiles?.find(
      (profile) => profile.id === agentProfiles.active_agent_profile_id,
    );
    if (
      !activeAgent ||
      activeAgent.name === WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME
    ) {
      return null;
    }
    if (activeAgent.agent_kind !== "openhands") {
      return t(I18nKey.SETTINGS$DEFAULT_LLM_PROFILE_AGENT_OWN_MODEL, {
        agent: activeAgent.name,
      });
    }
    // A ref to a missing LLM profile falls back to the default at launch.
    const ref = activeAgent.llm_profile_ref;
    if (
      !ref ||
      ref === activeProfile ||
      !profiles.some((p) => p.name === ref)
    ) {
      return null;
    }
    return t(I18nKey.SETTINGS$DEFAULT_LLM_PROFILE_AGENT_OVERRIDE, {
      agent: activeAgent.name,
      profile: ref,
    });
  }, [agentProfiles, activeProfile, profiles, t]);

  return (
    <section
      data-testid="default-llm-profile"
      aria-labelledby="default-llm-profile-title"
      className={cn(settingsListContainerClassName, "flex flex-col gap-3 p-4")}
    >
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-1.5">
          <h2
            id="default-llm-profile-title"
            className="text-base font-medium text-contrast"
          >
            {t(I18nKey.SETTINGS$DEFAULT_LLM_PROFILE_TITLE)}
          </h2>
          <StyledTooltip
            content={t(I18nKey.SETTINGS$DEFAULT_LLM_PROFILE_INFO)}
            placement="right"
            tooltipClassName="max-w-xs whitespace-pre-line text-left"
          >
            <button
              type="button"
              data-testid="default-llm-profile-info"
              aria-label={t(I18nKey.SETTINGS$DEFAULT_LLM_PROFILE_INFO_LABEL)}
              className="inline-flex cursor-help rounded-sm text-muted hover:text-contrast"
            >
              <Info aria-hidden className="h-4 w-4" />
            </button>
          </StyledTooltip>
        </div>
        <p className="text-sm text-muted">
          {t(I18nKey.SETTINGS$DEFAULT_LLM_PROFILE_DESCRIPTION)}
        </p>
      </div>
      <SettingsDropdownInput
        testId="default-llm-profile-select"
        name="default-llm-profile"
        ariaLabel={t(I18nKey.SETTINGS$DEFAULT_LLM_PROFILE_TITLE)}
        items={items}
        selectedKey={activeProfile ?? undefined}
        placeholder={t(I18nKey.SETTINGS$DEFAULT_LLM_PROFILE_PLACEHOLDER)}
        isClearable={false}
        isDisabled={!canManage || isActivating}
        onSelectionChange={(key) => {
          if (typeof key === "string" && key !== activeProfile) onSelect(key);
        }}
      />
      {selected?.description ? (
        <p
          data-testid="default-llm-profile-details"
          className="text-sm text-muted"
        >
          {selected.description}
        </p>
      ) : null}
      {overrideNote ? (
        <p
          data-testid="default-llm-profile-agent-note"
          className="text-sm text-muted"
        >
          {overrideNote}
        </p>
      ) : null}
    </section>
  );
}
