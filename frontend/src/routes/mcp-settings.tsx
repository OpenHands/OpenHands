import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { useSettings } from "#/hooks/query/use-settings";
import { useDeleteMcpServer } from "#/hooks/mutation/use-delete-mcp-server";
import { useAddMcpServer } from "#/hooks/mutation/use-add-mcp-server";
import { useUpdateMcpServer } from "#/hooks/mutation/use-update-mcp-server";
import { I18nKey } from "#/i18n/declaration";

import { MCPServerList } from "#/components/features/settings/mcp-settings/mcp-server-list";
import { MCPServerForm } from "#/components/features/settings/mcp-settings/mcp-server-form";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { BrandButton } from "#/components/features/settings/brand-button";
import { MCPConfig } from "#/types/settings";
import { parseMcpConfig } from "#/utils/mcp-config";
import { createPermissionGuard } from "#/utils/org/permission-guard";
import { Typography } from "#/ui/typography";

export const clientLoader = createPermissionGuard("manage_mcp");
export const handle = { hideTitle: true };

type MCPServerType = "sse" | "stdio" | "shttp";

interface MCPServerConfig {
  id: string;
  type: MCPServerType;
  name?: string;
  url?: string;
  api_key?: string;
  timeout?: number;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
}

function MCPSettingsScreen() {
  const { t } = useTranslation();
  const { data: settings, isLoading } = useSettings();
  const { mutate: deleteMcpServer } = useDeleteMcpServer();
  const { mutate: addMcpServer } = useAddMcpServer();
  const { mutate: updateMcpServer } = useUpdateMcpServer();

  const [view, setView] = useState<"list" | "add" | "edit">("list");
  const [editingServer, setEditingServer] = useState<MCPServerConfig | null>(
    null,
  );
  const [confirmationModalIsVisible, setConfirmationModalIsVisible] =
    useState(false);
  const [serverToDelete, setServerToDelete] = useState<string | null>(null);

  const mcpConfig: MCPConfig = parseMcpConfig(
    settings?.agent_settings?.mcp_config,
  );

  const allServers: MCPServerConfig[] = [
    ...mcpConfig.sse_servers.map((server, index) => ({
      id: `sse-${index}`,
      type: "sse" as const,
      url: typeof server === "string" ? server : server.url,
      api_key: typeof server === "object" ? server.api_key : undefined,
    })),
    ...mcpConfig.stdio_servers.map((server, index) => ({
      id: `stdio-${index}`,
      type: "stdio" as const,
      name: server.name,
      command: server.command,
      args: server.args,
      env: server.env,
    })),
    ...mcpConfig.shttp_servers.map((server, index) => ({
      id: `shttp-${index}`,
      type: "shttp" as const,
      url: typeof server === "string" ? server : server.url,
      api_key: typeof server === "object" ? server.api_key : undefined,
      timeout: typeof server === "object" ? server.timeout : undefined,
    })),
  ];

  const handleAddServer = (serverConfig: MCPServerConfig) => {
    addMcpServer(serverConfig, {
      onSuccess: () => {
        setView("list");
      },
    });
  };

  const handleEditServer = (serverConfig: MCPServerConfig) => {
    updateMcpServer(
      {
        serverId: serverConfig.id,
        server: serverConfig,
      },
      {
        onSuccess: () => {
          setView("list");
        },
      },
    );
  };

  const handleDeleteServer = (serverId: string) => {
    deleteMcpServer(serverId, {
      onSuccess: () => {
        setConfirmationModalIsVisible(false);
      },
    });
  };

  const handleEditClick = (server: MCPServerConfig) => {
    setEditingServer(server);
    setView("edit");
  };

  const handleDeleteClick = (serverId: string) => {
    setServerToDelete(serverId);
    setConfirmationModalIsVisible(true);
  };

  const handleConfirmDelete = () => {
    if (serverToDelete) {
      handleDeleteServer(serverToDelete);
    }
  };

  const handleCancelDelete = () => {
    setConfirmationModalIsVisible(false);
    setServerToDelete(null);
  };

  if (isLoading || !settings) {
    return null;
  }

  if (view === "add") {
    return (
      <MCPServerForm
        mode="add"
        existingServers={allServers}
        onSubmit={handleAddServer}
        onCancel={() => setView("list")}
      />
    );
  }

  if (view === "edit" && editingServer) {
    return (
      <MCPServerForm
        mode="edit"
        server={editingServer}
        existingServers={allServers}
        onSubmit={handleEditServer}
        onCancel={() => {
          setEditingServer(null);
          setView("list");
        }}
      />
    );
  }

  return (
    <div className="h-full max-w-[1000px] mx-auto flex flex-col px-6 gap-6 pb-8">
      <div className="flex justify-between items-center">
        <div>
          <Typography.H2 className="mb-2">
            {t(I18nKey.SETTINGS$MCP_TITLE)}
          </Typography.H2>
          <Typography.Paragraph className="text-sm text-[#A3A3A3]">
            {t(I18nKey.SETTINGS$MCP_DESCRIPTION)}
          </Typography.Paragraph>
        </div>
        <BrandButton
          type="button"
          variant="primary"
          onClick={() => setView("add")}
        >
          {t(I18nKey.SETTINGS$MCP_ADD_SERVER)}
        </BrandButton>
      </div>

      <MCPServerList
        servers={allServers}
        onEdit={handleEditClick}
        onDelete={handleDeleteClick}
      />

      {confirmationModalIsVisible && serverToDelete && (
        <ConfirmationModal
          text={t(I18nKey.SETTINGS$MCP_CONFIRM_DELETE)}
          onCancel={handleCancelDelete}
          onConfirm={handleConfirmDelete}
        />
      )}
    </div>
  );
}

export default MCPSettingsScreen;
