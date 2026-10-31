import { useState } from "react";
import { useTranslation } from "react-i18next";
import type { IntegrationCatalogEntry } from "@openhands/extensions/integrations";
import { BrandButton } from "#/components/features/settings/brand-button";
import {
  CustomServerEditor,
  InstalledServersSection,
  InstallServerModal,
  MarketplaceSection,
} from "#/components/features/mcp-page";
import { useUpdateMcpServer } from "#/hooks/mutation/use-update-mcp-server";
import type { MCPServerConfig } from "#/types/mcp-server";
import { I18nKey } from "#/i18n/declaration";

export function AgentMcpSetup({
  servers,
  onServerDeleted,
}: {
  servers: MCPServerConfig[];
  onServerDeleted?: (serverId: string) => void;
}) {
  const { t } = useTranslation("openhands");
  const { mutate: updateServer } = useUpdateMcpServer();
  const [installEntry, setInstallEntry] =
    useState<IntegrationCatalogEntry | null>(null);
  const [editingServer, setEditingServer] = useState<MCPServerConfig | null>(
    null,
  );

  return (
    <div data-testid="agent-mcp-setup" className="flex flex-col gap-4">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-medium text-foreground">
          {t(I18nKey.MCP$INSTALLED_TITLE)}
        </h3>
        <BrandButton
          type="button"
          variant="secondary"
          testId="agent-mcp-add-custom"
          onClick={() => setEditingServer({ id: "", type: "sse" })}
        >
          {t(I18nKey.MCP$ADD_CUSTOM)}
        </BrandButton>
      </div>
      <InstalledServersSection
        servers={servers}
        hasAnyInstalled={servers.length > 0}
        onEdit={setEditingServer}
        onToggleEnabled={(server, enabled) =>
          updateServer({ serverId: server.id, server: { ...server, enabled } })
        }
      />
      <div className="max-h-96 overflow-y-auto">
        <MarketplaceSection
          onSelect={setInstallEntry}
          onAdd={setInstallEntry}
        />
      </div>
      {installEntry && (
        <InstallServerModal
          entry={installEntry}
          existingServers={servers}
          mcpOnly
          onClose={() => setInstallEntry(null)}
        />
      )}
      {editingServer && (
        <CustomServerEditor
          server={editingServer}
          existingServers={servers}
          onClose={() => setEditingServer(null)}
          onDeleted={onServerDeleted}
        />
      )}
    </div>
  );
}
