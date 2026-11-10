import React from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Eye, EyeOff, Puzzle } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  INTEGRATION_CATALOG,
  type IntegrationCatalogEntry as MarketplaceEntry,
} from "@openhands/extensions/integrations";
import {
  changeConversationConfiguration,
  getConversationTools,
  type ConversationConfigurationOperation,
} from "#/api/conversation-configuration-service";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { MCPServerForm } from "#/components/features/settings/mcp-settings/mcp-server-form";
import { MarketplaceSection } from "#/components/features/mcp-page/marketplace-section";
import { InstallServerModal } from "#/components/features/mcp-page/install-server-modal";
import { McpLogoBadge } from "#/components/features/mcp-logo-badge";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import {
  ModalBody,
  MODAL_MAX_WIDTH_VIEWPORT,
} from "#/components/shared/modals/modal-body";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { NavigationLink } from "#/components/shared/navigation-link";
import { useSettings } from "#/hooks/query/use-settings";
import McpService from "#/api/mcp-service/mcp-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { CONVERSATION_CONFIGURATION_QUERY_KEYS } from "#/hooks/query/query-keys";
import { I18nKey } from "#/i18n/declaration";
import type { MCPServerConfig } from "#/types/mcp-server";
import {
  toCanonicalMcpServer,
  allocateMcpSettingsKey,
} from "#/utils/mcp-config";
import { formControlSettingsFieldClassName } from "#/utils/form-control-classes";
import { getInstalledServerTitle } from "#/utils/mcp-installed-server-display";
import { flattenMcpConfig } from "#/utils/mcp-installed-servers";
import {
  findCatalogEntryForServer,
  getMcpMarketplaceCatalog,
} from "#/utils/mcp-marketplace-utils";

interface ConversationConfigurationModalProps {
  conversationId: string;
  isAcp: boolean;
  canInstallMcp?: boolean;
  canListTools?: boolean;
  onClose: () => void;
}

type AppliedChange = {
  type: "success" | "error";
  name?: string;
};

function SavedMcpSection({
  onAttach,
  pending,
  attached,
  canInstallMcp,
}: {
  onAttach: (ref: string) => void;
  pending: boolean;
  attached: Set<string>;
  canInstallMcp: boolean;
}) {
  const { t } = useTranslation("openhands");
  const { data: settings, isLoading, isError, refetch } = useSettings();
  const catalog = React.useMemo(
    () => getMcpMarketplaceCatalog(INTEGRATION_CATALOG),
    [],
  );
  const servers = flattenMcpConfig(settings?.mcp_config ?? {}).filter(
    (server) => server.enabled !== false,
  );

  return (
    <section
      data-testid="conversation-mcp-section"
      className="w-full space-y-3"
    >
      <h3 className="text-sm font-semibold text-contrast">
        {t(I18nKey.MCP$INSTALLED_TITLE)}
      </h3>
      {isLoading ? (
        <p className="text-sm text-muted">{t(I18nKey.HOME$LOADING)}</p>
      ) : isError ? (
        <BrandButton
          variant="secondary"
          type="button"
          onClick={() => void refetch()}
        >
          {t(I18nKey.MCP$HEALTH_RETRY)}
        </BrandButton>
      ) : servers.length === 0 ? (
        <p className="text-sm text-muted">
          {t(I18nKey.MCP$INSTALLED_EMPTY_TITLE)}
        </p>
      ) : (
        <ul className="max-h-44 space-y-2 overflow-y-auto">
          {servers.map((server) => {
            const entry = findCatalogEntryForServer(server, catalog);
            const title = getInstalledServerTitle(server, entry);
            return (
              <li
                key={server.id}
                data-testid={`conversation-mcp-${server.id}`}
                className="flex items-center gap-3 rounded-lg border border-border p-2"
              >
                <McpLogoBadge
                  entry={entry}
                  size="base"
                  fallback={<Puzzle size={16} />}
                />
                <span className="min-w-0 flex-1 truncate text-sm" title={title}>
                  {title}
                </span>
                <BrandButton
                  variant="secondary"
                  type="button"
                  isDisabled={pending || attached.has(server.id)}
                  ariaLabel={t(I18nKey.MCP$ATTACH_TO_CONVERSATION, {
                    name: title,
                  })}
                  onClick={() => onAttach(server.id)}
                >
                  {t(I18nKey.BUTTON$ADD)}
                </BrandButton>
              </li>
            );
          })}
        </ul>
      )}
      {!canInstallMcp && (
        <NavigationLink
          to="/mcp"
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm text-primary underline"
        >
          {t(I18nKey.MCP$VIEW_MARKETPLACE)}
        </NavigationLink>
      )}
    </section>
  );
}

export function ConversationConfigurationModal({
  conversationId,
  isAcp,
  canInstallMcp = false,
  canListTools = false,
  onClose,
}: ConversationConfigurationModalProps) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const { backend } = useActiveBackend();
  const { data: settings } = useSettings();
  const [installEntry, setInstallEntry] =
    React.useState<MarketplaceEntry | null>(null);
  const [editingCustom, setEditingCustom] = React.useState(false);
  const [marketplaceSearch, setMarketplaceSearch] = React.useState("");
  const [name, setName] = React.useState("");
  const [value, setValue] = React.useState("");
  const [showValue, setShowValue] = React.useState(false);
  const [pending, setPending] = React.useState(false);
  const [result, setResult] = React.useState<AppliedChange | null>(null);
  const [attached, setAttached] = React.useState(() => new Set<string>());
  const { data: liveTools } = useQuery({
    queryKey: CONVERSATION_CONFIGURATION_QUERY_KEYS.tools(
      backend.id,
      backend.connectionRevision,
      conversationId,
    ),
    queryFn: () => getConversationTools(conversationId),
    enabled: !isAcp && canListTools,
    retry: false,
    meta: { disableToast: true },
  });
  const attachedNames = new Set([
    ...attached,
    ...(liveTools?.tools.flatMap((tool) =>
      tool.source === "conversation" && tool.server_name
        ? [tool.server_name]
        : [],
    ) ?? []),
  ]);
  const valueId = React.useId();

  const applyChange = async (
    operation: ConversationConfigurationOperation,
    label: string,
  ) => {
    setPending(true);
    setResult(null);
    try {
      await changeConversationConfiguration(conversationId, {
        operations: [operation],
      });
      if (operation.type === "secret.grant") {
        setName("");
        setValue("");
        setShowValue(false);
      } else {
        setAttached((current) =>
          new Set(current).add(
            operation.type === "mcp.attach" ? operation.ref : operation.name,
          ),
        );
        void queryClient.invalidateQueries({
          queryKey: CONVERSATION_CONFIGURATION_QUERY_KEYS.tools(
            backend.id,
            backend.connectionRevision,
            conversationId,
          ),
        });
      }
      setResult({ type: "success", name: label });
      return true;
    } catch {
      setResult({ type: "error" });
      return false;
    } finally {
      setPending(false);
    }
  };

  const installMcp = async (server: MCPServerConfig) => {
    const preferred = allocateMcpSettingsKey(
      settings?.mcp_config ?? {},
      server.name,
      server.type,
    );
    let name = preferred;
    for (let suffix = 1; attachedNames.has(name); suffix += 1) {
      name = `${preferred}_${suffix}`;
    }
    const installed = await applyChange(
      { type: "mcp.install", name, server: toCanonicalMcpServer(server) },
      name,
    );
    if (!installed) throw new Error("MCP install failed");
  };

  const installCustomMcp = async (server: MCPServerConfig) => {
    try {
      let configured = server;
      if (server.auth?.strategy === "oauth2") {
        setPending(true);
        const result = await McpService.authorizeOAuth(server);
        if (!result.ok) {
          setResult({ type: "error" });
          return;
        }
        if (result.oauth_state) {
          configured = {
            ...server,
            auth: { ...server.auth, state: result.oauth_state },
          };
        }
      }
      await installMcp(configured);
      setEditingCustom(false);
    } catch {
      setResult({ type: "error" });
    } finally {
      setPending(false);
    }
  };

  const handleSecretSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmedName = name.trim();
    if (!trimmedName || !value.trim() || pending) return;
    void applyChange(
      { type: "secret.grant", name: trimmedName, value },
      trimmedName,
    );
  };

  if (installEntry) {
    return (
      <InstallServerModal
        entry={installEntry}
        existingServers={[]}
        onInstall={installMcp}
        onClose={() => setInstallEntry(null)}
      />
    );
  }

  return (
    <ModalBackdrop
      onClose={() => {
        if (!pending) onClose();
      }}
      aria-label={t(I18nKey.CHAT$CONVERSATION_CONFIGURATION)}
    >
      <ModalBody
        width="md"
        testID="conversation-configuration-modal"
        className={`relative dialog-content-height overflow-y-auto ${MODAL_MAX_WIDTH_VIEWPORT}`}
      >
        <ModalCloseButton onClose={onClose} disabled={pending} />
        <div className="w-full space-y-2 pr-8">
          <h2 className="text-lg font-semibold text-contrast">
            {t(I18nKey.CHAT$CONVERSATION_CONFIGURATION)}
          </h2>
          <p className="text-sm text-muted">
            {t(I18nKey.CHAT$CONVERSATION_CONFIGURATION_SCOPE)}
          </p>
        </div>
        <form onSubmit={handleSecretSubmit} className="w-full space-y-3">
          <h3 className="text-sm font-semibold text-contrast">
            {t(I18nKey.SECRETS$ADD_SECRET)}
          </h3>
          <SettingsInput
            testId="conversation-secret-name"
            name="secret-name"
            type="text"
            label={t(I18nKey.SETTINGS$NAME)}
            value={name}
            onChange={setName}
            required
            pattern="[a-zA-Z][a-zA-Z0-9_]{0,63}"
            title={t(I18nKey.SETTINGS$SECRET_NAME_PATTERN_TITLE)}
          />
          <div className="w-full space-y-2">
            <label htmlFor={valueId} className="text-sm">
              {t(I18nKey.FORM$VALUE)}
            </label>
            <div className="flex items-center gap-2">
              <input
                id={valueId}
                name="secret-value"
                data-testid="conversation-secret-value"
                type={showValue ? "text" : "password"}
                autoComplete="new-password"
                required
                value={value}
                onChange={(event) => setValue(event.currentTarget.value)}
                className={formControlSettingsFieldClassName}
              />
              <BrandButton
                type="button"
                variant="secondary"
                ariaLabel={t(
                  showValue
                    ? I18nKey.SECRETS$HIDE_VALUE
                    : I18nKey.SECRETS$SHOW_VALUE,
                )}
                onClick={() => setShowValue((visible) => !visible)}
              >
                {showValue ? (
                  <EyeOff size={16} aria-hidden />
                ) : (
                  <Eye size={16} aria-hidden />
                )}
              </BrandButton>
            </div>
          </div>
          <BrandButton
            type="submit"
            variant="primary"
            isDisabled={pending}
            aria-busy={pending}
          >
            {t(I18nKey.SECRETS$ADD_SECRET)}
          </BrandButton>
        </form>
        {!isAcp && (
          <SavedMcpSection
            onAttach={(ref) =>
              void applyChange({ type: "mcp.attach", ref }, ref)
            }
            pending={pending}
            attached={attachedNames}
            canInstallMcp={canInstallMcp}
          />
        )}
        {!isAcp && canInstallMcp && (
          <section className="w-full space-y-3">
            {editingCustom ? (
              <MCPServerForm
                mode="add"
                onSubmit={(server) => void installCustomMcp(server)}
                onCancel={() => setEditingCustom(false)}
                isActionDisabled={pending}
              />
            ) : (
              <>
                <SettingsInput
                  testId="conversation-mcp-marketplace-search"
                  name="marketplace-search"
                  type="search"
                  label={t(I18nKey.MCP$SEARCH_PLACEHOLDER)}
                  value={marketplaceSearch}
                  onChange={setMarketplaceSearch}
                />
                <div className="max-h-64 overflow-y-auto">
                  <MarketplaceSection
                    query={marketplaceSearch}
                    onSelect={setInstallEntry}
                    onAdd={setInstallEntry}
                  />
                </div>
                <BrandButton
                  type="button"
                  variant="secondary"
                  onClick={() => setEditingCustom(true)}
                  isDisabled={pending}
                >
                  {t(I18nKey.MCP$ADD_CUSTOM)}
                </BrandButton>
              </>
            )}
          </section>
        )}
        {result?.type === "success" && (
          <p role="status" className="w-full text-sm text-feedback-success">
            {t(I18nKey.CHAT$CONVERSATION_CONFIGURATION_SUCCESS, {
              name: result.name,
            })}
          </p>
        )}
        {result?.type === "error" && (
          <p role="alert" className="w-full text-sm text-feedback-error">
            {t(I18nKey.CHAT$CONVERSATION_CONFIGURATION_FAILED)}
          </p>
        )}
      </ModalBody>
    </ModalBackdrop>
  );
}
