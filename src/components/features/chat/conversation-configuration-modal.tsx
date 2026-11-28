import React from "react";
import { Eye, EyeOff, Puzzle } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  INTEGRATION_CATALOG,
  type IntegrationCatalogEntry as MarketplaceEntry,
} from "@openhands/extensions/integrations";
import type { MCPConfig, MCPServer } from "@openhands/typescript-client";
import {
  addConversationMcpServers,
  selectConversationMcpServers,
  updateConversationSecrets,
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
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { useLlmProfiles } from "#/hooks/query/use-llm-profiles";
import { useSaveLlmProfile } from "#/hooks/mutation/use-save-llm-profile";
import { useSwitchLlmProfile } from "#/hooks/mutation/use-switch-llm-profile";
import { isProfileNameValid } from "#/utils/derive-profile-name";
import type { SaveProfileRequest } from "#/api/profiles-service/profiles-service.api";

export type ConversationConfigurationSection = "secrets" | "mcp" | "models";

interface ConversationConfigurationModalProps {
  conversationId: string;
  section: ConversationConfigurationSection;
  onClose: () => void;
}

type AppliedChange = {
  type: "success" | "error";
  name?: string;
  message?: string;
};

function SavedMcpSection({
  onAttach,
  pending,
  attached,
}: {
  onAttach: (refs: string[]) => void;
  pending: boolean;
  attached: Set<string>;
}) {
  const { t } = useTranslation("openhands");
  const { data: settings, isLoading, isError, refetch } = useSettings();
  const [selected, setSelected] = React.useState(() => new Set<string>());
  const catalog = React.useMemo(
    () => getMcpMarketplaceCatalog(INTEGRATION_CATALOG),
    [],
  );
  const servers = flattenMcpConfig(settings?.mcp_config ?? {}).filter(
    (server) => server.enabled !== false && server.type !== "sse",
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
    </section>
  );
}

export function ConversationConfigurationModal({
  conversationId,
  section,
  onClose,
}: ConversationConfigurationModalProps) {
  const { t } = useTranslation("openhands");
  const { data: settings } = useSettings();
  const { data: llmProfiles } = useLlmProfiles();
  const saveProfile = useSaveLlmProfile();
  const switchProfile = useSwitchLlmProfile();
  const [profileName, setProfileName] = React.useState("");
  const [model, setModel] = React.useState("");
  const [apiKey, setApiKey] = React.useState("");
  const [baseUrl, setBaseUrl] = React.useState("");
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

  const applySecret = async (secretName: string, secretValue: string) => {
    setPending(true);
    setResult(null);
    try {
      await updateConversationSecrets(conversationId, {
        secrets: { [secretName]: secretValue },
      });
      setName("");
      setValue("");
      setShowValue(false);
      setResult({ type: "success", name: secretName });
      return true;
    } catch {
      setResult({ type: "error" });
      return false;
    } finally {
      setPending(false);
    }
  };

  const applyMcpServers = async (
    mcpServers: MCPConfig,
    labels: string[],
    options: { throwOnError?: boolean } = {},
  ) => {
    setPending(true);
    setResult(null);
    try {
      await addConversationMcpServers(conversationId, {
        mcp_servers: mcpServers,
      });
      setAttached((current) => {
        const next = new Set(current);
        for (const label of labels) next.add(label);
        return next;
      });
      setResult({ type: "success", name: labels.join(", ") });
      return true;
    } catch (err: unknown) {
      const message = retrieveAxiosErrorMessage(err);
      setResult({ type: "error", ...(message ? { message } : {}) });
      if (options.throwOnError) throw err;
      return false;
    } finally {
      setPending(false);
    }
  };

  const attachSelected = async (refs: string[]) => {
    const mcpConfig = settings?.mcp_config as MCPConfig | undefined;
    for (const ref of refs) {
      const mcpServers = selectConversationMcpServers(mcpConfig, [ref]);
      if (Object.keys(mcpServers).length === 0) {
        setResult({ type: "error" });
        break;
      }
      if (!(await applyMcpServers(mcpServers, [ref]))) break;
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
    await applyMcpServers(
      { [name]: toCanonicalMcpServer(server) as MCPServer },
      [name],
      { throwOnError: true },
    );
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
    void applySecret(trimmedName, value);
  };

  const handleProfileSubmit = async (
    event: React.FormEvent<HTMLFormElement>,
  ) => {
    event.preventDefault();
    const trimmedName = profileName.trim();
    if (
      !isProfileNameValid(trimmedName, { isRequired: true }) ||
      !model.trim() ||
      llmProfiles?.profiles.some((profile) => profile.name === trimmedName) ||
      pending
    )
      return;

    setPending(true);
    setResult(null);
    try {
      const llm: SaveProfileRequest["llm"] = {
        model: model.trim(),
        stream: true,
        ...(apiKey ? { api_key: apiKey } : {}),
        ...(baseUrl.trim() ? { base_url: baseUrl.trim() } : {}),
      };
      await saveProfile.mutateAsync({ name: trimmedName, request: { llm } });
      await switchProfile.mutateAsync({
        conversationId,
        profileName: trimmedName,
      });
      setApiKey("");
      setResult({ type: "success", name: trimmedName });
    } catch (error) {
      setResult({ type: "error", message: retrieveAxiosErrorMessage(error) });
    } finally {
      setPending(false);
    }
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
      aria-label={t(
        section === "secrets"
          ? I18nKey.SECRETS$ADD_SECRET
          : section === "mcp"
            ? I18nKey.MCP$INSTALL_BUTTON
            : I18nKey.SETTINGS$ADD_LLM_PROFILE,
      )}
    >
      <ModalBody
        width="md"
        testID="conversation-configuration-modal"
        className={`relative dialog-content-height overflow-y-auto ${MODAL_MAX_WIDTH_VIEWPORT}`}
      >
        <ModalCloseButton onClose={onClose} disabled={pending} />
        <div className="w-full space-y-2 pr-8">
          <h2 className="text-lg font-semibold text-contrast">
            {t(
              section === "secrets"
                ? I18nKey.SECRETS$ADD_SECRET
                : section === "mcp"
                  ? I18nKey.MCP$INSTALL_BUTTON
                  : I18nKey.SETTINGS$ADD_LLM_PROFILE,
            )}
          </h2>
          {section !== "models" && (
            <p className="text-sm text-muted">
              {t(I18nKey.CHAT$CONVERSATION_CONFIGURATION_SCOPE)}
            </p>
          )}
        </div>
        {section === "secrets" && (
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
        )}
        {section === "mcp" && (
          <SavedMcpSection
            onAttach={(refs) => void attachSelected(refs)}
            pending={pending}
            attached={attached}
          />
        )}
        {section === "mcp" && (
          <section className="w-full space-y-3">
            {editingCustom ? (
              <MCPServerForm
                mode="add"
                runtimeOnly
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
        {section === "models" && (
          <section className="w-full space-y-4">
            <h3 className="text-sm font-semibold text-contrast">
              {t(I18nKey.SETTINGS$AVAILABLE_PROFILES)}
            </h3>
            <ul className="max-h-44 space-y-2 overflow-y-auto">
              {llmProfiles?.profiles.map((profile) => (
                <li key={profile.name}>
                  <BrandButton
                    type="button"
                    variant="secondary"
                    isDisabled={pending}
                    onClick={() => {
                      setPending(true);
                      setResult(null);
                      void switchProfile
                        .mutateAsync({
                          conversationId,
                          profileName: profile.name,
                        })
                        .then(() =>
                          setResult({ type: "success", name: profile.name }),
                        )
                        .catch((error: unknown) =>
                          setResult({
                            type: "error",
                            message: retrieveAxiosErrorMessage(error),
                          }),
                        )
                        .finally(() => setPending(false));
                    }}
                  >
                    {profile.name} · {profile.model}
                  </BrandButton>
                </li>
              ))}
            </ul>
            <form
              onSubmit={(event) => void handleProfileSubmit(event)}
              className="space-y-3"
            >
              <h3 className="text-sm font-semibold text-contrast">
                {t(I18nKey.SETTINGS$ADD_LLM_PROFILE)}
              </h3>
              <SettingsInput
                testId="conversation-llm-name"
                name="profile-name"
                type="text"
                label={t(I18nKey.SETTINGS$PROFILE_NAME_LABEL)}
                value={profileName}
                onChange={setProfileName}
                required
              />
              <SettingsInput
                testId="conversation-llm-model"
                name="model"
                type="text"
                label={t(I18nKey.SCHEMA$LLM$MODEL$LABEL)}
                value={model}
                onChange={setModel}
                required
              />
              <SettingsInput
                testId="conversation-llm-api-key"
                name="api-key"
                type="password"
                label={t(I18nKey.SCHEMA$LLM$API_KEY$LABEL)}
                value={apiKey}
                onChange={setApiKey}
              />
              <SettingsInput
                testId="conversation-llm-base-url"
                name="base-url"
                type="url"
                label={t(I18nKey.SETTINGS$BASE_URL)}
                value={baseUrl}
                onChange={setBaseUrl}
              />
              <BrandButton
                type="submit"
                variant="primary"
                isDisabled={
                  pending ||
                  !isProfileNameValid(profileName.trim(), {
                    isRequired: true,
                  }) ||
                  !model.trim() ||
                  !!llmProfiles?.profiles.some(
                    (profile) => profile.name === profileName.trim(),
                  )
                }
              >
                {t(I18nKey.SETTINGS$ADD_LLM_PROFILE)}
              </BrandButton>
            </form>
            <NavigationLink
              to="/settings/llm"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-primary underline"
            >
              {t(I18nKey.SETTINGS$AVAILABLE_PROFILES)}
            </NavigationLink>
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
            {result.message ||
              t(I18nKey.CHAT$CONVERSATION_CONFIGURATION_FAILED)}
          </p>
        )}
      </ModalBody>
    </ModalBackdrop>
  );
}
