import React from "react";
import { useTranslation } from "react-i18next";
import { ExtensionsNavigation } from "#/components/features/skills/extensions-navigation";
import { PluginCard } from "#/components/features/plugins/plugin-card";
import { PluginsToolbar } from "#/components/features/plugins/plugins-toolbar";
import { PluginDetailModal } from "#/components/features/plugins/plugin-detail-modal";
import { AddPluginModal } from "#/components/features/plugins/add-plugin-modal";
import { BrandButton } from "#/components/features/settings/brand-button";
import {
  buildPluginsViewModel,
  matchesPluginSearch,
  matchesPluginStatus,
  type PluginStatusFilter,
  type PluginViewModel,
} from "#/components/features/plugins/build-plugins-view-model";
import { usePluginsMarketplace } from "#/hooks/query/use-plugins-marketplace";
import { usePlugins } from "#/hooks/query/use-plugins";
import { useLocalPlugins } from "#/hooks/query/use-local-plugins";
import { useInstallPlugin } from "#/hooks/mutation/use-install-plugin";
import { useSetPluginEnabled } from "#/hooks/mutation/use-set-plugin-enabled";
import { useUninstallPlugin } from "#/hooks/mutation/use-uninstall-plugin";
import { useRefreshPlugin } from "#/hooks/mutation/use-refresh-plugin";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useNavigation } from "#/context/navigation-context";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { buildPluginLaunchPath } from "#/utils/plugin-launch-url";
import { settingsLikeMainScrollClassName } from "#/utils/settings-like-page-layout-classes";
import {
  extensionModuleCardGridClassName,
  extensionModuleCardGridContainerClassName,
  extensionModuleEmptyStateClassName,
} from "#/utils/extension-module-card-classes";

export default function SkillsPluginsScreen() {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const { navigate } = useNavigation();
  const isLocal = backend.kind === "local";

  const { data: marketplace, isLoading: marketplaceLoading } =
    usePluginsMarketplace();
  const { data: installed, isLoading: installedLoading } = usePlugins();
  const { data: local, isLoading: localLoading } = useLocalPlugins();

  const installPlugin = useInstallPlugin();
  const setPluginEnabled = useSetPluginEnabled();
  const uninstallPlugin = useUninstallPlugin();
  const refreshPlugin = useRefreshPlugin();

  const [searchQuery, setSearchQuery] = React.useState("");
  const [statusFilter, setStatusFilter] =
    React.useState<PluginStatusFilter>("all");
  const [selectedName, setSelectedName] = React.useState<string | null>(null);
  const [showAddModal, setShowAddModal] = React.useState(false);

  const plugins = React.useMemo(
    () => buildPluginsViewModel(marketplace, installed, local),
    [marketplace, installed, local],
  );

  const filteredPlugins = React.useMemo(
    () =>
      plugins.filter(
        (plugin) =>
          matchesPluginStatus(plugin, statusFilter) &&
          matchesPluginSearch(plugin, searchQuery),
      ),
    [plugins, statusFilter, searchQuery],
  );

  const selectedPlugin = selectedName
    ? (plugins.find((plugin) => plugin.name === selectedName) ?? null)
    : null;

  const isLoading = marketplaceLoading || installedLoading || localLoading;

  const pendingName =
    (setPluginEnabled.isPending
      ? setPluginEnabled.variables?.name
      : undefined) ??
    (uninstallPlugin.isPending ? uninstallPlugin.variables : undefined) ??
    (refreshPlugin.isPending ? refreshPlugin.variables : undefined) ??
    null;

  const isPluginBusy = (plugin: PluginViewModel): boolean =>
    pendingName === plugin.name ||
    (installPlugin.isPending &&
      installPlugin.variables?.source === plugin.source);

  const handleInstall = (plugin: PluginViewModel) => {
    if (!plugin.source) return;
    installPlugin.mutate({
      source: plugin.source,
      ref: plugin.ref,
      repo_path: plugin.repoPath,
    });
  };

  const handleToggle = (plugin: PluginViewModel, enabled: boolean) => {
    setPluginEnabled.mutate({ name: plugin.name, enabled });
  };

  const handleUninstall = (plugin: PluginViewModel) => {
    uninstallPlugin.mutate(plugin.name, {
      onSuccess: () => setSelectedName(null),
    });
  };

  const handleRefresh = (plugin: PluginViewModel) => {
    refreshPlugin.mutate(plugin.name);
  };

  const handleStartConversation = (plugin: PluginViewModel) => {
    if (!plugin.source) return;
    navigate(
      buildPluginLaunchPath([
        { source: plugin.source, ref: plugin.ref, repo_path: plugin.repoPath },
      ]),
    );
  };

  return (
    <div
      data-testid="skills-plugins-screen"
      className="flex h-full gap-4 md:gap-6 md:pl-8 lg:gap-10 lg:pl-10"
    >
      <ExtensionsNavigation />
      <main className={cn(settingsLikeMainScrollClassName, "h-full")}>
        <div className="mx-auto flex w-full min-w-0 max-w-[800px] flex-col gap-6">
          <div className="flex min-w-0 items-start justify-between gap-4">
            <div className="min-w-0 space-y-1">
              <h2 className="text-xl font-semibold leading-6 text-foreground">
                {t(I18nKey.SETTINGS$PLUGINS_TITLE)}
              </h2>
              <div
                data-testid="plugins-settings-description"
                className="max-w-2xl text-sm text-tertiary-light"
              >
                {t(I18nKey.SETTINGS$PLUGINS_PAGE_DESCRIPTION)}
              </div>
            </div>
            <BrandButton
              type="button"
              variant="secondary"
              testId="plugins-add-plugin-button"
              isDisabled={!isLocal}
              className="flex-shrink-0 whitespace-nowrap"
              onClick={() => setShowAddModal(true)}
            >
              {t(I18nKey.SETTINGS$PLUGINS_ADD_BUTTON)}
            </BrandButton>
          </div>

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

          {!isLoading && plugins.length === 0 && (
            <div
              data-testid="plugins-empty"
              className={extensionModuleEmptyStateClassName}
            >
              <p className="text-sm text-tertiary-light">
                {t(I18nKey.SETTINGS$PLUGINS_NO_PLUGINS)}
              </p>
            </div>
          )}

          {!isLoading && plugins.length > 0 && (
            <>
              <PluginsToolbar
                search={searchQuery}
                onSearchChange={setSearchQuery}
                statusFilter={statusFilter}
                onStatusFilterChange={setStatusFilter}
              />
              {filteredPlugins.length === 0 ? (
                <div
                  data-testid="plugins-no-match"
                  className={extensionModuleEmptyStateClassName}
                >
                  <p className="text-sm text-tertiary-light">
                    {t(I18nKey.SETTINGS$PLUGINS_NO_MATCH)}
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
                    {filteredPlugins.map((plugin) => (
                      <PluginCard
                        key={plugin.name}
                        plugin={plugin}
                        isBusy={isPluginBusy(plugin)}
                        isDisabled={!isLocal}
                        onOpen={() => setSelectedName(plugin.name)}
                        onInstall={() => handleInstall(plugin)}
                        onToggle={(enabled) => handleToggle(plugin, enabled)}
                      />
                    ))}
                  </div>
                </section>
              )}
            </>
          )}

          {selectedPlugin && (
            <PluginDetailModal
              plugin={selectedPlugin}
              isBusy={isPluginBusy(selectedPlugin)}
              isDisabled={!isLocal}
              onToggle={(enabled) => handleToggle(selectedPlugin, enabled)}
              onInstall={() => handleInstall(selectedPlugin)}
              onUninstall={() => handleUninstall(selectedPlugin)}
              onRefresh={() => handleRefresh(selectedPlugin)}
              onStartConversation={() =>
                handleStartConversation(selectedPlugin)
              }
              onClose={() => setSelectedName(null)}
            />
          )}

          {showAddModal && (
            <AddPluginModal onClose={() => setShowAddModal(false)} />
          )}
        </div>
      </main>
    </div>
  );
}
