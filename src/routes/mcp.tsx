import React from "react";
import { AxiosError } from "axios";
import { ExtensionsNavigation } from "#/components/features/skills/extensions-navigation";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { useSettings } from "#/hooks/query/use-settings";
import { useDeleteMcpServer } from "#/hooks/mutation/use-delete-mcp-server";
import { parseMcpConfig } from "#/utils/mcp-config";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { settingsLikeMainScrollClassName } from "#/utils/settings-like-page-layout-classes";
import {
  findCatalogEntryForServer,
  getMcpMarketplaceCatalog,
  installedServerMatchesQuery,
} from "#/utils/mcp-marketplace-utils";
import {
  INTEGRATION_CATALOG as MCP_MARKETPLACE,
  type IntegrationCatalogEntry as MarketplaceEntry,
} from "@openhands/extensions/integrations";
import { MCPServerConfig } from "#/types/mcp-server";
import { flattenMcpConfig } from "#/utils/mcp-installed-servers";
import {
  InstalledServersSection,
  McpToolbar,
  MarketplaceSection,
  InstallServerModal,
  CustomServerEditor,
  type McpSectionFilter,
} from "#/components/features/mcp-page";

// No ACP guard here (unlike `/settings` and `/settings/condenser`): MCP
// servers configured via `agent_settings.mcp_config` are now forwarded to
// the ACP subprocess at session creation, so this page is meaningful for
// both OpenHands and ACP agents. The same editor and MCP settings storage
// drive both kinds.

export default function MCPPage() {
  const { t } = useTranslation("openhands");
  const { data: settings, isLoading } = useSettings();
  const { mutate: deleteMcpServer, isPending: isDeleting } =
    useDeleteMcpServer();

  const [installEntry, setInstallEntry] =
    React.useState<MarketplaceEntry | null>(null);
  const [editingServer, setEditingServer] =
    React.useState<MCPServerConfig | null>(null);
  const [serverToDelete, setServerToDelete] =
    React.useState<MCPServerConfig | null>(null);
  const [searchQuery, setSearchQuery] = React.useState("");
  const [sectionFilter, setSectionFilter] =
    React.useState<McpSectionFilter>("all");

  const mcpConfig = parseMcpConfig(settings?.agent_settings?.mcp_config);
  const allServers = flattenMcpConfig(mcpConfig);
  const mcpMarketplace = getMcpMarketplaceCatalog(MCP_MARKETPLACE);

  // Filter installed servers by the search query. We pair each server
  // with its catalog entry (if any) so the search can match friendly
  // names like "Slack" against a stdio server whose own `.name` is
  // just "slack".
  const filteredInstalledServers = allServers.filter((server) =>
    installedServerMatchesQuery(
      server,
      findCatalogEntryForServer(server, mcpMarketplace),
      searchQuery,
    ),
  );

  const handleMarketplaceInstall = (entry: MarketplaceEntry) => {
    setInstallEntry(entry);
  };

  const handleEdit = (server: MCPServerConfig) => {
    setEditingServer(server);
  };

  const handleDeleteClick = (serverId: string) => {
    const target = allServers.find((s) => s.id === serverId);
    if (target) setServerToDelete(target);
  };

  const handleConfirmDelete = () => {
    if (!serverToDelete) return;
    // Pass the full server config — useDeleteMcpServer re-resolves its
    // position against the fresh settings at mutation time, so a
    // background refresh between this click and confirm cannot point
    // us at the wrong index.
    deleteMcpServer(serverToDelete, {
      onSuccess: () => {
        displaySuccessToast(t(I18nKey.MCP$REMOVE_SUCCESS));
        setServerToDelete(null);
      },
      onError: (err) => {
        const message = retrieveAxiosErrorMessage(err as AxiosError);
        displayErrorToast(message || t(I18nKey.ERROR$GENERIC));
        setServerToDelete(null);
      },
    });
  };

  if (isLoading || !settings) {
    return (
      <div
        data-testid="mcp-page"
        className="flex h-full gap-4 md:gap-6 md:pl-8 lg:gap-10 lg:pl-10"
      >
        <ExtensionsNavigation />
        <div className="flex h-full flex-1 items-center justify-center px-4 md:px-0">
          <div className="h-8 w-8 rounded-full border-2 border-transparent border-t-white animate-spin" />
        </div>
      </div>
    );
  }

  return (
    <div
      data-testid="mcp-page"
      className="flex h-full gap-4 md:gap-6 md:pl-8 lg:gap-10 lg:pl-10"
    >
      <ExtensionsNavigation />
      <main className={settingsLikeMainScrollClassName}>
        <div className="mx-auto flex w-full min-w-0 max-w-[800px] flex-col gap-6">
          <div className="min-w-0">
            <div className="flex items-start justify-between gap-4">
              <div className="space-y-1">
                <h2 className="text-xl font-medium leading-6 text-foreground">
                  {t(I18nKey.SETTINGS$MCP_TITLE)}
                </h2>
                <div className="max-w-2xl text-sm text-tertiary-light">
                  {t(I18nKey.MCP$PAGE_DESCRIPTION)}
                </div>
              </div>
              <BrandButton
                type="button"
                variant="secondary"
                testId="mcp-add-custom-server"
                className="flex-shrink-0 whitespace-nowrap"
                onClick={() => setEditingServer({ id: "", type: "sse" })}
              >
                {t(I18nKey.MCP$ADD_CUSTOM)}
              </BrandButton>
            </div>
          </div>

          <McpToolbar
            search={searchQuery}
            onSearchChange={setSearchQuery}
            sectionFilter={sectionFilter}
            onSectionFilterChange={setSectionFilter}
          />

          {sectionFilter !== "library" ? (
            <section className="flex flex-col gap-3">
              <h2 className="text-base font-semibold text-foreground">
                {t(I18nKey.MCP$INSTALLED_TITLE)}
              </h2>
              <InstalledServersSection
                servers={filteredInstalledServers}
                hasAnyInstalled={allServers.length > 0}
                query={searchQuery}
                onEdit={handleEdit}
                onDelete={handleDeleteClick}
              />
            </section>
          ) : null}

          {sectionFilter !== "installed" ? (
            <MarketplaceSection
              onSelect={handleMarketplaceInstall}
              onAdd={handleMarketplaceInstall}
              query={searchQuery}
            />
          ) : null}
        </div>

        {installEntry && (
          <InstallServerModal
            entry={installEntry}
            existingServers={allServers}
            onClose={() => setInstallEntry(null)}
          />
        )}

        {/* Custom (or non-marketplace) server editor. The empty-id
            sentinel (`{ id: "", type: "sse" }`) means "add new". */}
        {editingServer && (
          <CustomServerEditor
            server={editingServer}
            existingServers={allServers}
            onClose={() => setEditingServer(null)}
          />
        )}

        {serverToDelete && (
          <ConfirmationModal
            text={t(I18nKey.SETTINGS$MCP_CONFIRM_DELETE)}
            onCancel={() => setServerToDelete(null)}
            onConfirm={handleConfirmDelete}
            isConfirming={isDeleting}
          />
        )}
      </main>
    </div>
  );
}
