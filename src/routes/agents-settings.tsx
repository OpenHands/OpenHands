import React from "react";
import { useTranslation } from "react-i18next";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useSettings } from "#/hooks/query/use-settings";
import { useAgents } from "#/hooks/query/use-agents";
import { useNavigation } from "#/context/navigation-context";
import { ExtensionsNavigation } from "#/components/features/skills/extensions-navigation";
import { AgentCard } from "#/components/features/agents/agent-card";
import { AgentDetailModal } from "#/components/features/agents/agent-detail-modal";
import { AddAgentModal } from "#/components/features/agents/add-agent-modal";
import { AgentsToolbar } from "#/components/features/agents/agents-toolbar";
import { BrandButton } from "#/components/features/settings/brand-button";
import type { AgentLevelFilter } from "#/components/features/agents/agent-level-filter";
import { I18nKey } from "#/i18n/declaration";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { cn } from "#/utils/utils";
import { settingsLikeMainScrollClassName } from "#/utils/settings-like-page-layout-classes";
import {
  extensionModuleCardGridClassName,
  extensionModuleCardGridContainerClassName,
  extensionModuleEmptyStateClassName,
} from "#/utils/extension-module-card-classes";
import type { AgentInfo } from "#/types/settings";
import { getAgentCardDescription } from "#/components/features/agents/get-agent-card-description";

function matchesSearch(agent: AgentInfo, query: string): boolean {
  if (!query) return true;
  const haystacks = [
    agent.name,
    getAgentCardDescription(agent),
    agent.description ?? "",
    agent.system_prompt ?? "",
    agent.model ?? "",
    ...(agent.tools ?? []),
    ...(agent.when_to_use_examples ?? []),
  ];
  const lowered = query.toLowerCase();
  return haystacks.some((value) => value.toLowerCase().includes(lowered));
}

function AgentsSettingsScreen() {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();

  const { mutate: saveSettings } = useSaveSettings();
  const { data: settings, isLoading: settingsLoading } = useSettings();
  const { data: agents, isLoading: agentsLoading } = useAgents();

  const [disabledSet, setDisabledSet] = React.useState<Set<string>>(new Set());
  const [hasHydratedInitialSettings, setHasHydratedInitialSettings] =
    React.useState(false);
  const [searchQuery, setSearchQuery] = React.useState("");
  const [levelFilter, setLevelFilter] = React.useState<AgentLevelFilter>("all");
  const [selectedAgent, setSelectedAgent] = React.useState<AgentInfo | null>(
    null,
  );
  const [showAddAgentModal, setShowAddAgentModal] = React.useState(false);

  // Sync local state with server settings when data first arrives
  React.useEffect(() => {
    if (settingsLoading || !settings) return;
    setDisabledSet(new Set(settings.disabled_agents ?? []));
    setHasHydratedInitialSettings(true);
  }, [settingsLoading, settings?.disabled_agents]);

  const handleToggle = (agentName: string, enabled: boolean) => {
    setDisabledSet((prev) => {
      const next = new Set(prev);
      if (enabled) {
        next.delete(agentName);
      } else {
        next.add(agentName);
      }
      return next;
    });
  };

  // Auto-save agent toggles once initial settings are loaded.
  React.useEffect(() => {
    if (!hasHydratedInitialSettings) return;
    saveSettings(
      { disabled_agents: Array.from(disabledSet) },
      {
        onError: (error) => {
          const errorMessage = retrieveAxiosErrorMessage(error);
          displayErrorToast(errorMessage || t(I18nKey.ERROR$GENERIC));
        },
      },
    );
  }, [disabledSet, hasHydratedInitialSettings, saveSettings, t]);

  const isLoading = settingsLoading || agentsLoading || !settings;

  const subAgentsEnabled =
    (settings?.agent_settings?.enable_sub_agents ?? false) === true;

  const filteredAgents = React.useMemo(() => {
    if (!agents) return [];
    return agents.filter(
      (agent) =>
        (levelFilter === "all" || agent.level === levelFilter) &&
        matchesSearch(agent, searchQuery),
    );
  }, [agents, levelFilter, searchQuery]);

  return (
    <div
      data-testid="agents-settings-screen"
      className="flex h-full gap-4 md:gap-6 md:pl-8 lg:gap-10 lg:pl-10"
    >
      <ExtensionsNavigation />
      <main className={cn(settingsLikeMainScrollClassName, "h-full")}>
        <div className="mx-auto flex w-full min-w-0 max-w-[800px] flex-col gap-6">
          <div className="flex min-w-0 items-start justify-between gap-4">
            <div className="min-w-0 space-y-1">
              <h2 className="text-xl font-semibold leading-6 text-foreground">
                {t(I18nKey.SETTINGS$AGENTS_TITLE)}
              </h2>
              <div
                data-testid="agents-settings-description"
                className="max-w-2xl text-sm text-tertiary-light"
              >
                {t(I18nKey.SETTINGS$AGENTS_PAGE_DESCRIPTION)}
              </div>
            </div>
            <BrandButton
              type="button"
              variant="secondary"
              testId="agents-add-agent-button"
              className="flex-shrink-0 whitespace-nowrap"
              onClick={() => setShowAddAgentModal(true)}
            >
              {t(I18nKey.SETTINGS$AGENTS_ADD_BUTTON)}
            </BrandButton>
          </div>

          {!isLoading && !subAgentsEnabled && agents && agents.length > 0 && (
            <div
              data-testid="agents-subagents-hint"
              className="rounded-lg border border-[var(--oh-border)] bg-[rgba(255,255,255,0.04)] px-3 py-2.5 text-xs leading-relaxed text-tertiary-light"
            >
              {t(I18nKey.SETTINGS$AGENTS_SUBAGENTS_HINT)}{" "}
              <button
                type="button"
                data-testid="agents-subagents-hint-cta"
                onClick={() => navigate("/settings/agent")}
                className="cursor-pointer text-white underline hover:no-underline"
              >
                {t(I18nKey.SETTINGS$AGENTS_SUBAGENTS_HINT_CTA)}
              </button>
            </div>
          )}

          {isLoading && (
            <div className="flex flex-col gap-4">
              {[1, 2, 3].map((i) => (
                <div
                  key={i}
                  className="h-24 rounded-2xl bg-tertiary animate-pulse"
                />
              ))}
            </div>
          )}

          {!isLoading && (!agents || agents.length === 0) && (
            <div
              data-testid="agents-empty"
              className={extensionModuleEmptyStateClassName}
            >
              <p className="text-sm text-tertiary-light">
                {t(I18nKey.SETTINGS$AGENTS_NO_AGENTS)}
              </p>
            </div>
          )}

          {!isLoading && agents && agents.length > 0 && (
            <>
              <AgentsToolbar
                search={searchQuery}
                onSearchChange={setSearchQuery}
                levelFilter={levelFilter}
                onLevelFilterChange={setLevelFilter}
              />
              {filteredAgents.length === 0 ? (
                <div
                  data-testid="agents-no-match"
                  className={extensionModuleEmptyStateClassName}
                >
                  <p className="text-sm text-tertiary-light">
                    {t(I18nKey.SETTINGS$AGENTS_NO_MATCH)}
                  </p>
                </div>
              ) : (
                <section
                  className={cn(
                    "flex min-w-0 flex-col gap-3",
                    extensionModuleCardGridContainerClassName,
                  )}
                >
                  <div className={extensionModuleCardGridClassName}>
                    {filteredAgents.map((agent) => (
                      <AgentCard
                        key={agent.name}
                        agent={agent}
                        enabled={!disabledSet.has(agent.name)}
                        onOpen={() => setSelectedAgent(agent)}
                        onToggle={(enabled) =>
                          handleToggle(agent.name, enabled)
                        }
                      />
                    ))}
                  </div>
                </section>
              )}
            </>
          )}

          {selectedAgent && (
            <AgentDetailModal
              agent={selectedAgent}
              enabled={!disabledSet.has(selectedAgent.name)}
              onToggle={(enabled) => handleToggle(selectedAgent.name, enabled)}
              onClose={() => setSelectedAgent(null)}
            />
          )}

          {showAddAgentModal && (
            <AddAgentModal onClose={() => setShowAddAgentModal(false)} />
          )}
        </div>
      </main>
    </div>
  );
}

export default AgentsSettingsScreen;
