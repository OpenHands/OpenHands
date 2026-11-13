import React from "react";
import { Eye, EyeOff, Puzzle } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  INTEGRATION_CATALOG,
  type IntegrationCatalogEntry as MarketplaceEntry,
} from "@openhands/extensions/integrations";
import {
  changeConversationConfiguration,
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
  onAttach: (refs: string[]) => void;
  pending: boolean;
  attached: Set<string>;
  canInstallMcp: boolean;
}) {
  const { t } = useTranslation("openhands");
  const { data: settings, isLoading, isError, refetch } = useSettings();
  const [selected, setSelected] = React.useState(() => new Set<string>());
  const catalog = React.useMemo(
    () => getMcpMarketplaceCatalog(INTEGRATION_CATALOG),
    [],
  );
  const servers = flattenMcpConfig(settings?.mcp_config ?? {}).filter(
    (server) => server.enabled !== false,
  );
  const pendingRefs = servers
    .filter((server) => selected.has(server.id) && !attached.has(server.id))
    .map((server) => server.id);

  const toggle = (ref: string, checked: boolean) => {
    setSelected((current) => {
      const next = new Set(current);
      if (checked) next.add(ref);
      else next.delete(ref);
      return next;
    });
  };

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
        <>
          <ul className="max-h-44 space-y-2 overflow-y-auto">
            {servers.map((server) => {
              const entry = findCatalogEntryForServer(server, catalog);
              const title = getInstalledServerTitle(server, entry);
              return (
                <li
                  key={server.id}
                  data-testid={`conversation-mcp-${server.id}`}
                >
                  <label className="flex items-center gap-3 rounded-lg border border-border p-2">
                    <input
                      type="checkbox"
                      className="size-4 shrink-0 accent-primary"
                      checked={
                        attached.has(server.id) || selected.has(server.id)
                      }
                      disabled={pending || attached.has(server.id)}
                      onChange={(event) =>
                        toggle(server.id, event.target.checked)
                      }
                      aria-label={t(I18nKey.MCP$ATTACH_TO_CONVERSATION, {
                        name: title,
                      })}
                    />
                    <McpLogoBadge
                      entry={entry}
                      size="base"
                      fallback={<Puzzle size={16} />}
                    />
                    <span
                      className="min-w-0 flex-1 truncate text-sm"
                      title={title}
                    >
                      {title}
                    </span>
                  </label>
                </li>
              );
            })}
          </ul>
          <BrandButton
            testId="conversation-mcp-add-selected"
            variant="secondary"
            type="button"
            isDisabled={pending || pendingRefs.length === 0}
            onClick={() => onAttach(pendingRefs)}
          >
            {t(I18nKey.BUTTON$ADD)} ({pendingRefs.length})
          </BrandButton>
        </>
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
  onClose,
}: ConversationConfigurationModalProps) {
  const { t } = useTranslation("openhands");
  const { data: settings } = useSettings();
  const [installQueue, setInstallQueue] = React.useState<MarketplaceEntry[]>(
    [],
  );
  const [selectedMarketplace, setSelectedMarketplace] = React.useState(
    () => new Set<string>(),
  );
  const [editingCustom, setEditingCustom] = React.useState(false);
  const [marketplaceSearch, setMarketplaceSearch] = React.useState("");
  const [name, setName] = React.useState("");
  const [value, setValue] = React.useState("");
  const [showValue, setShowValue] = React.useState(false);
  const [pending, setPending] = React.useState(false);
  const [result, setResult] = React.useState<AppliedChange | null>(null);
  const [attached, setAttached] = React.useState(() => new Set<string>());
  const catalog = React.useMemo(
    () => getMcpMarketplaceCatalog(INTEGRATION_CATALOG),
    [],
  );
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

  const attachSelected = async (refs: string[]) => {
    for (const ref of refs) {
      if (!(await applyChange({ type: "mcp.attach", ref }, ref))) break;
    }
  };

  const installMcp = async (server: MCPServerConfig) => {
    const preferred = allocateMcpSettingsKey(
      settings?.mcp_config ?? {},
      server.name,
      server.type,
    );
    let name = preferred;
    for (let suffix = 1; attached.has(name); suffix += 1) {
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

  if (installQueue.length > 0) {
    const entry = installQueue[0];
    return (
      <InstallServerModal
        key={entry.id}
        entry={entry}
        existingServers={[]}
        onInstall={installMcp}
        onSuccess={() =>
          setSelectedMarketplace((current) => {
            const next = new Set(current);
            next.delete(entry.id);
            return next;
          })
        }
        onClose={() => setInstallQueue((current) => current.slice(1))}
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
            onAttach={(refs) => void attachSelected(refs)}
            pending={pending}
            attached={attached}
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
                    onSelect={(entry) => setInstallQueue([entry])}
                    onAdd={(entry) => setInstallQueue([entry])}
                    selectedEntryIds={selectedMarketplace}
                    onToggleEntry={(entry, checked) =>
                      setSelectedMarketplace((current) => {
                        const next = new Set(current);
                        if (checked) next.add(entry.id);
                        else next.delete(entry.id);
                        return next;
                      })
                    }
                  />
                </div>
                <div className="flex flex-wrap gap-2">
                  <BrandButton
                    testId="conversation-mcp-install-selected"
                    type="button"
                    variant="primary"
                    isDisabled={pending || selectedMarketplace.size === 0}
                    onClick={() =>
                      setInstallQueue(
                        catalog.filter((entry) =>
                          selectedMarketplace.has(entry.id),
                        ),
                      )
                    }
                  >
                    {t(I18nKey.MCP$INSTALL_BUTTON)} ({selectedMarketplace.size})
                  </BrandButton>
                  <BrandButton
                    type="button"
                    variant="secondary"
                    onClick={() => setEditingCustom(true)}
                    isDisabled={pending}
                  >
                    {t(I18nKey.MCP$ADD_CUSTOM)}
                  </BrandButton>
                </div>
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
