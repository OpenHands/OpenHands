import React, { useEffect, useState } from "react";
import { AxiosError } from "axios";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { useSettings } from "#/hooks/query/use-settings";
import { useDeleteMcpServer } from "#/hooks/mutation/use-delete-mcp-server";
import { useUpdateMcpServer } from "#/hooks/mutation/use-update-mcp-server";
import { parseMcpConfig } from "#/utils/mcp-config";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
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
  CustomServerEditor,
  InstallServerModal,
  MarketplaceSection,
  McpToolbar,
  type McpSectionFilter,
} from "#/components/features/mcp-page";
import { useConversationOverviewDrawerOptional } from "./conversation-overview-drawer-context";

interface ConversationOverviewMcpPanelProps {
  openAdd: boolean;
}

export function ConversationOverviewMcpPanel({
  openAdd,
}: ConversationOverviewMcpPanelProps) {
  const { t } = useTranslation("openhands");
  const { data: settings, isLoading } = useSettings();
  const { mutate: deleteMcpServer, isPending: isDeleting } =
    useDeleteMcpServer();
  const { mutate: updateMcpServer } = useUpdateMcpServer();
  const addRequestKey =
    useConversationOverviewDrawerOptional()?.addRequestKey ?? 0;

  const [installEntry, setInstallEntry] = useState<MarketplaceEntry | null>(
    null,
  );
  const [editingServer, setEditingServer] = useState<MCPServerConfig | null>(
    null,
  );
  const [serverToDelete, setServerToDelete] = useState<MCPServerConfig | null>(
    null,
  );
  const [searchQuery, setSearchQuery] = useState("");
  const [sectionFilter, setSectionFilter] = useState<McpSectionFilter>("all");

  useEffect(() => {
    if (!openAdd) {
      return;
    }
    setEditingServer({ id: "", type: "sse" });
  }, [openAdd]);

  useEffect(() => {
    if (addRequestKey === 0) {
      return;
    }
    setEditingServer({ id: "", type: "sse" });
  }, [addRequestKey]);

  const mcpConfig = parseMcpConfig(settings?.agent_settings?.mcp_config);
  const allServers = flattenMcpConfig(mcpConfig);
  const mcpMarketplace = getMcpMarketplaceCatalog(MCP_MARKETPLACE);
  const filteredInstalledServers = allServers.filter((server) =>
    installedServerMatchesQuery(
      server,
      findCatalogEntryForServer(server, mcpMarketplace),
      searchQuery,
    ),
  );

  const handleToggleEnabled = (server: MCPServerConfig, enabled: boolean) => {
    updateMcpServer({ serverId: server.id, server: { ...server, enabled } });
  };

  const handleConfirmDelete = () => {
    if (!serverToDelete) {
      return;
    }
    deleteMcpServer(serverToDelete, {
      onSuccess: () => {
        displaySuccessToast(t(I18nKey.MCP$REMOVE_SUCCESS));
        setServerToDelete(null);
      },
      onError: (error) => {
        displayErrorToast(
          retrieveAxiosErrorMessage(error as AxiosError) ||
            t(I18nKey.ERROR$GENERIC),
        );
        setServerToDelete(null);
      },
    });
  };

  if (isLoading || !settings) {
    return (
      <div
        data-testid="conversation-overview-mcp-panel"
        className="text-sm text-muted"
      >
        …
      </div>
    );
  }

  return (
    <div
      data-testid="conversation-overview-mcp-panel"
      className="flex flex-col gap-4"
    >
      <McpToolbar
        search={searchQuery}
        onSearchChange={setSearchQuery}
        sectionFilter={sectionFilter}
        onSectionFilterChange={setSectionFilter}
      />

      {sectionFilter !== "library" ? (
        <section className="flex flex-col gap-3">
          <h3 className="text-sm font-semibold text-foreground">
            {t(I18nKey.MCP$INSTALLED_TITLE)}
          </h3>
          <InstalledServersSection
            servers={filteredInstalledServers}
            hasAnyInstalled={allServers.length > 0}
            query={searchQuery}
            onEdit={setEditingServer}
            onToggleEnabled={handleToggleEnabled}
          />
        </section>
      ) : null}

      {sectionFilter !== "installed" ? (
        <MarketplaceSection
          onSelect={setInstallEntry}
          onAdd={setInstallEntry}
          query={searchQuery}
        />
      ) : null}

      {installEntry ? (
        <InstallServerModal
          entry={installEntry}
          existingServers={allServers}
          onClose={() => setInstallEntry(null)}
        />
      ) : null}

      {editingServer ? (
        <CustomServerEditor
          server={editingServer}
          existingServers={allServers}
          onClose={() => setEditingServer(null)}
        />
      ) : null}

      {serverToDelete ? (
        <ConfirmationModal
          text={t(I18nKey.SETTINGS$MCP_CONFIRM_DELETE)}
          onCancel={() => setServerToDelete(null)}
          onConfirm={handleConfirmDelete}
          isConfirming={isDeleting}
        />
      ) : null}
    </div>
  );
}
