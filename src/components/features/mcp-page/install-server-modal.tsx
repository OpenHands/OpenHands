import React from "react";
import { useTranslation } from "react-i18next";
import { AxiosError } from "axios";
import { v4 as uuidv4 } from "uuid";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { SaveAsSecretToggle } from "#/components/features/mcp-page/save-as-secret-toggle";
import { I18nKey } from "#/i18n/declaration";
import type {
  IntegrationCatalogEntry as MarketplaceEntry,
  MarketplaceField,
} from "@openhands/extensions/integrations";
import { McpLogoBadge } from "#/components/features/mcp-logo-badge";
import { MCPServerConfig } from "#/types/mcp-server";
import type { MCPAuthCredential } from "#/types/mcp-auth";
import { seedMcpServerHealth } from "#/api/mcp-health/probe-mcp-server-health";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useAddMcpServer } from "#/hooks/mutation/use-add-mcp-server";
import { useTestMcpServer } from "#/hooks/mutation/use-test-mcp-server";
import { displaySuccessToast } from "#/utils/custom-toast-handlers";
import {
  getMcpOAuthAuthenticationConfig,
  getInstallableMcpConnectionOption,
  type McpMarketplaceConnectionOption,
} from "#/utils/mcp-marketplace-utils";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { useSaveFieldsAsSecrets } from "#/hooks/mutation/use-save-fields-as-secrets";
import { makeMcpTestErrorMessage } from "#/utils/mcp-test-error-message";
import { modalTitleLgClassName } from "#/utils/modal-classes";
import McpService from "#/api/mcp-service/mcp-service.api";
import { toMcpServerName } from "#/utils/mcp-server-name";

/**
 * Renders a helperText string as React nodes, converting any `[text](url)`
 * markdown links into real `<a>` elements. Plain text segments are left as-is.
 * Only `http:` and `https:` URLs are rendered as links; anything else falls
 * back to `#` to guard against `javascript:` / `data:` XSS vectors.
 */
function renderHelperText(text: string): React.ReactNode {
  const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g;
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  for (const match of text.matchAll(linkPattern)) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    parts.push(
      <a
        key={match.index}
        href={/^https?:\/\//i.test(match[2]) ? match[2] : "#"}
        target="_blank"
        rel="noreferrer"
        className="underline hover:text-white transition-colors"
      >
        {match[1]}
      </a>,
    );
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return parts;
}

interface InstallServerModalProps {
  entry: MarketplaceEntry;
  /** Servers installed before this modal opened — guards health seeding. */
  existingServers: MCPServerConfig[];
  onClose: () => void;
  onSuccess?: (entry: MarketplaceEntry) => void;
}

interface FieldState {
  values: Record<string, string>;
  errors: Record<string, string | null>;
  savedAsSecret: Record<string, boolean>;
}

function optionNeedsCredentialField(
  option: McpMarketplaceConnectionOption | undefined,
): boolean {
  if (option?.transport.kind !== "shttp" && option?.transport.kind !== "sse") {
    return false;
  }
  return ["api_key", "bearer"].includes(option.auth.strategy);
}

function isOAuthOption(
  option: McpMarketplaceConnectionOption | undefined,
): boolean {
  return !!option && option.auth.strategy === "oauth2";
}

function isCredentialOptional(option: McpMarketplaceConnectionOption): boolean {
  if (option.transport.kind === "stdio") {
    return option.auth.apiKeyOptional ?? false;
  }
  return option.auth.apiKeyOptional ?? option.transport.apiKeyOptional ?? false;
}

function getRemoteHeaderFields(
  option: McpMarketplaceConnectionOption | undefined,
): MarketplaceField[] {
  if (option?.transport.kind !== "shttp" && option?.transport.kind !== "sse") {
    return [];
  }
  return option.transport.headerFields ?? [];
}

function makeInitialState(entry: MarketplaceEntry): FieldState {
  const values: Record<string, string> = {};
  const savedAsSecret: Record<string, boolean> = {};
  const option = getInstallableMcpConnectionOption(entry);
  const template = option?.transport;
  if (template?.kind === "stdio") {
    for (const field of template.envFields ?? []) {
      values[field.key] = "";
      // Pre-check password fields; non-password fields default to off.
      savedAsSecret[field.key] = field.type === "password";
    }
    for (const field of template.argFields ?? []) {
      values[field.key] = "";
    }
  } else if (template?.kind === "shttp" || template?.kind === "sse") {
    values.url = template.url;
    for (const field of getRemoteHeaderFields(option)) {
      values[field.key] = "";
      savedAsSecret[field.key] = field.type === "password";
    }
    if (optionNeedsCredentialField(option)) {
      values.api_key = "";
      if (option?.auth.credentialSecretName) {
        savedAsSecret.api_key =
          option.auth.saveCredentialAsSecretByDefault ?? false;
      }
    }
  }
  return { values, errors: {}, savedAsSecret };
}

// The marketplace install modal is intentionally add-only: clicking
// a catalog tile always appends a new server (the user might want
// two Slack workspaces, two Postgres connections, etc.) even when
// one of the same template kind is already installed. Editing an
// existing server is reached via the installed-server-card's edit
// button, which opens `CustomServerEditor` instead.
export function InstallServerModal({
  entry,
  existingServers,
  onClose,
  onSuccess,
}: InstallServerModalProps) {
  const { t } = useTranslation("openhands");
  const { mutate: addMcpServer, isPending: isAdding } = useAddMcpServer();
  const { mutate: testMcpServer, isPending: isTesting } = useTestMcpServer();
  const saveFieldsAsSecrets = useSaveFieldsAsSecrets();
  // Cloud backends get a synthetic test success (no local test endpoint);
  // never seed card health from it.
  const { backend } = useActiveBackend();
  const isCloudBackend = backend.kind === "cloud";

  const [state, setState] = React.useState<FieldState>(() =>
    makeInitialState(entry),
  );
  // Always holds the latest state so async callbacks (onSuccess) never read
  // stale closure values, even under React concurrent-mode scheduling.
  const stateRef = React.useRef(state);
  stateRef.current = state;

  const [globalError, setGlobalError] = React.useState<string | null>(null);
  const [isFinalizingInstall, setIsFinalizingInstall] = React.useState(false);
  const [isAuthorizingOAuth, setIsAuthorizingOAuth] = React.useState(false);
  const option = getInstallableMcpConnectionOption(entry);
  const template = option?.transport;

  const isPending =
    isTesting || isAuthorizingOAuth || isAdding || isFinalizingInstall;

  const setValue = (key: string, value: string) => {
    setState((prev) => ({
      ...prev,
      values: { ...prev.values, [key]: value },
      errors: { ...prev.errors, [key]: null },
    }));
    setGlobalError(null);
  };

  const toggleSecret = (key: string, value: boolean) => {
    setState((prev) => ({
      ...prev,
      savedAsSecret: { ...prev.savedAsSecret, [key]: value },
    }));
  };

  const saveHostedCredentialAsSecret = (): Promise<void> => {
    const secretName = option?.auth.credentialSecretName;
    const apiKey = stateRef.current.values.api_key?.trim();
    if (!secretName || !apiKey || !stateRef.current.savedAsSecret.api_key) {
      return Promise.resolve();
    }

    const field: MarketplaceField = {
      key: secretName,
      label: option.auth.credentialLabel ?? secretName,
      type: "password",
      required: true,
    };
    return saveFieldsAsSecrets(
      [field],
      { [secretName]: apiKey },
      { [secretName]: true },
    );
  };

  const saveSelectedSecrets = (): Promise<void> => {
    if (template?.kind === "stdio") {
      return saveFieldsAsSecrets(
        template.envFields ?? [],
        stateRef.current.values,
        stateRef.current.savedAsSecret,
      );
    }
    if (template?.kind === "shttp" || template?.kind === "sse") {
      return Promise.all([
        saveHostedCredentialAsSecret(),
        saveFieldsAsSecrets(
          getRemoteHeaderFields(option),
          stateRef.current.values,
          stateRef.current.savedAsSecret,
        ),
      ]).then(() => undefined);
    }
    return Promise.resolve();
  };

  const submitServer = (payload: MCPServerConfig) => {
    if (payload.auth?.strategy === "oauth2") {
      setIsAuthorizingOAuth(true);
      void McpService.authorizeOAuth(payload)
        .then((result) => {
          if (!result.ok) {
            setGlobalError(
              makeMcpTestErrorMessage(t, result.error_kind, result.error),
            );
            return;
          }
          const serverToSave = result.oauth_state
            ? {
                ...payload,
                auth: { ...payload.auth!, state: result.oauth_state },
              }
            : payload;
          addMcpServer(serverToSave, {
            onSuccess: () => {
              if (!isCloudBackend) {
                seedMcpServerHealth(serverToSave, result, existingServers);
              }
              displaySuccessToast(t(I18nKey.MCP$INSTALL_SUCCESS));
              setIsFinalizingInstall(true);
              void (async () => {
                try {
                  await saveSelectedSecrets();
                } finally {
                  onSuccess?.(entry);
                  onClose();
                }
              })();
            },
            onError: (err: unknown) => {
              const message = retrieveAxiosErrorMessage(err as AxiosError);
              setGlobalError(message || t(I18nKey.ERROR$GENERIC));
            },
          });
        })
        .catch((err: unknown) => {
          const message = retrieveAxiosErrorMessage(err as AxiosError);
          setGlobalError(message || t(I18nKey.ERROR$GENERIC));
        })
        .finally(() => setIsAuthorizingOAuth(false));
      return;
    }
    testMcpServer(payload, {
      onSuccess: (result) => {
        if (!result.ok) {
          setGlobalError(
            makeMcpTestErrorMessage(t, result.error_kind, result.error),
          );
          // Modal stays open — do NOT call onClose.
          return;
        }
        const serverToSave =
          result.oauth_state && payload.auth?.strategy === "oauth2"
            ? {
                ...payload,
                auth: { ...payload.auth, state: result.oauth_state },
              }
            : payload;
        addMcpServer(serverToSave, {
          onSuccess: () => {
            if (!isCloudBackend) {
              seedMcpServerHealth(serverToSave, result, existingServers);
            }
            displaySuccessToast(t(I18nKey.MCP$INSTALL_SUCCESS));
            setIsFinalizingInstall(true);
            void (async () => {
              try {
                await saveSelectedSecrets();
              } finally {
                onSuccess?.(entry);
                onClose();
              }
            })();
          },
          onError: (err: unknown) => {
            const message = retrieveAxiosErrorMessage(err as AxiosError);
            setGlobalError(message || t(I18nKey.ERROR$GENERIC));
          },
        });
      },
      onError: (err: unknown) => {
        const message = retrieveAxiosErrorMessage(err as AxiosError);
        setGlobalError(message || t(I18nKey.ERROR$GENERIC));
      },
    });
  };

  // ------------------------------------------------------------------
  // Per-template submit handlers. Each is small and self-contained:
  // validate user input, build the payload, then hand off to
  // submitServer.
  // ------------------------------------------------------------------
  const handleHttpServerSubmit = () => {
    // TS narrows this branch to shttp|sse; the equality guard is a
    // runtime/defensive belt to make the helper safe in isolation.
    if (template?.kind !== "shttp" && template?.kind !== "sse") {
      return;
    }
    if (!option) return;
    const apiKey = state.values.api_key?.trim() ?? "";
    const url = template.urlEditable
      ? (state.values.url?.trim() ?? "")
      : template.url;
    const oauthMode = isOAuthOption(option);
    const needsCredential = optionNeedsCredentialField(option);
    const headerFields = getRemoteHeaderFields(option);
    const headerErrors: Record<string, string | null> = {};
    if (!url) {
      headerErrors.url = t(I18nKey.SETTINGS$MCP_ERROR_URL_REQUIRED);
    } else {
      try {
        const parsedUrl = new URL(url);
        if (!["http:", "https:"].includes(parsedUrl.protocol)) {
          headerErrors.url = t(I18nKey.SETTINGS$MCP_ERROR_URL_INVALID_PROTOCOL);
        }
      } catch {
        headerErrors.url = t(I18nKey.SETTINGS$MCP_ERROR_URL_INVALID);
      }
    }
    for (const field of headerFields) {
      if (field.required && !(state.values[field.key] ?? "").trim()) {
        headerErrors[field.key] = t(I18nKey.MCP$ERROR_FIELD_REQUIRED);
      }
    }
    if (
      !oauthMode &&
      needsCredential &&
      !isCredentialOptional(option) &&
      !apiKey
    ) {
      headerErrors.api_key = t(I18nKey.MCP$ERROR_FIELD_REQUIRED);
    }
    if (Object.values(headerErrors).some(Boolean)) {
      setState((prev) => ({ ...prev, errors: headerErrors }));
      return;
    }
    const oauthAuthentication = oauthMode
      ? getMcpOAuthAuthenticationConfig(option)
      : undefined;
    const fieldHeaders = Object.fromEntries(
      headerFields
        .map((field) => [field.key, state.values[field.key]?.trim() ?? ""])
        .filter(([, value]) => value),
    );
    const hasFieldHeaders = Object.keys(fieldHeaders).length > 0;
    let auth: MCPAuthCredential | undefined;
    if (oauthMode) {
      auth = {
        strategy: "oauth2",
        ...(oauthAuthentication && { authentication: oauthAuthentication }),
      };
    } else if (needsCredential && apiKey) {
      auth =
        option.auth.strategy === "api_key"
          ? {
              strategy: "api_key",
              value: apiKey,
              ...(option.auth.apiKeyHeaderName && {
                header_name: option.auth.apiKeyHeaderName,
              }),
            }
          : { strategy: "bearer", value: apiKey };
    } else if (hasFieldHeaders) {
      auth = { strategy: "header", headers: fieldHeaders };
    }
    const payload: MCPServerConfig = {
      id: `${template.kind}-${uuidv4()}`,
      type: template.kind,
      // Name remote servers after the catalog slug (e.g. "github") so they
      // get a referenceable, LLM-tool-safe mcp_config key instead of the
      // auto-generated "sse"/"shttp" fallback. Stdio installs already carry
      // serverName from the catalog.
      name: toMcpServerName(entry.id),
      url,
      ...(auth && { auth }),
      ...(hasFieldHeaders &&
        auth?.strategy !== "header" && { headers: fieldHeaders }),
    };
    submitServer(payload);
  };

  const handleStdioSubmit = () => {
    if (template?.kind !== "stdio") return;
    const stdio = template;
    const errors: Record<string, string | null> = {};

    for (const field of stdio.envFields ?? []) {
      if (field.required && !(state.values[field.key] ?? "").trim()) {
        errors[field.key] = t(I18nKey.MCP$ERROR_FIELD_REQUIRED);
      }
    }
    for (const field of stdio.argFields ?? []) {
      if (field.required && !(state.values[field.key] ?? "").trim()) {
        errors[field.key] = t(I18nKey.MCP$ERROR_FIELD_REQUIRED);
      }
    }
    if (Object.values(errors).some(Boolean)) {
      setState((prev) => ({ ...prev, errors }));
      return;
    }

    const env: Record<string, string> = {};
    for (const field of stdio.envFields ?? []) {
      const v = state.values[field.key]?.trim();
      if (v) env[field.key] = v;
    }
    const extraArgs: string[] = [];
    for (const field of stdio.argFields ?? []) {
      const v = state.values[field.key]?.trim();
      if (v) {
        // Filesystem-style multi-token input: split on whitespace.
        for (const token of v.split(/\s+/)) {
          if (token) extraArgs.push(token);
        }
      }
    }

    const payload: MCPServerConfig = {
      id: `stdio-${uuidv4()}`,
      type: "stdio",
      name: stdio.serverName,
      command: stdio.command,
      args: [...stdio.args, ...extraArgs],
      ...(Object.keys(env).length > 0 && { env }),
    };
    submitServer(payload);
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setGlobalError(null);
    if (template?.kind === "shttp" || template?.kind === "sse") {
      return handleHttpServerSubmit();
    }
    return handleStdioSubmit();
  };

  const renderFields = () => {
    if (template?.kind === "shttp" || template?.kind === "sse") {
      const oauthMode = isOAuthOption(option);
      const shouldRenderCredential = optionNeedsCredentialField(option);
      const apiKeyOptional = option ? isCredentialOptional(option) : false;
      const credentialSecretName = option?.auth.credentialSecretName;
      const headerFields = getRemoteHeaderFields(option);
      return (
        <>
          <SettingsInput
            testId="mcp-install-field-url"
            name="url"
            type="url"
            label={t(I18nKey.SETTINGS$MCP_URL)}
            value={state.values.url ?? template.url}
            onChange={(value) => setValue("url", value)}
            isDisabled={!template.urlEditable}
            className="w-full"
          />
          {state.errors.url && (
            <p className="text-xs text-red-500">{state.errors.url}</p>
          )}
          {headerFields.map((field) => (
            <div key={field.key} className="flex flex-col gap-1">
              <SettingsInput
                testId={`mcp-install-field-${field.key}`}
                name={field.key}
                type={field.type === "password" ? "password" : "text"}
                label={field.label}
                value={state.values[field.key] ?? ""}
                onChange={(v) => setValue(field.key, v)}
                placeholder={field.placeholder}
                required={field.required}
                showOptionalTag={!field.required}
                className="w-full"
              />
              {field.helperText && (
                <p className="text-xs text-tertiary-alt">
                  {renderHelperText(field.helperText)}
                </p>
              )}
              {state.errors[field.key] && (
                <p className="text-xs text-red-500">
                  {state.errors[field.key]}
                </p>
              )}
              {field.key in state.savedAsSecret && (
                <SaveAsSecretToggle
                  fieldKey={field.key}
                  checked={state.savedAsSecret[field.key] ?? false}
                  onToggle={(v) => toggleSecret(field.key, v)}
                />
              )}
            </div>
          ))}
          {oauthMode ? (
            <div
              data-testid="mcp-install-oauth-info"
              className="flex flex-col gap-2 p-3 rounded-lg border border-[var(--oh-border)] bg-base-tertiary"
            >
              <p className="text-sm text-secondary-light">
                {t(I18nKey.MCP$OAUTH_CONNECT_INFO)}
              </p>
              <p className="text-xs text-tertiary-alt">
                {t(I18nKey.MCP$OAUTH_CONNECT_HINT)}
              </p>
            </div>
          ) : shouldRenderCredential ? (
            <div className="flex flex-col gap-1">
              <SettingsInput
                testId="mcp-install-field-api_key"
                name="api_key"
                type="password"
                label={
                  option?.auth.credentialLabel ??
                  t(I18nKey.SETTINGS$MCP_API_KEY)
                }
                value={state.values.api_key ?? ""}
                onChange={(v) => setValue("api_key", v)}
                placeholder={
                  option?.auth.credentialPlaceholder ??
                  t(I18nKey.SETTINGS$MCP_API_KEY_PLACEHOLDER)
                }
                showOptionalTag={apiKeyOptional}
                required={!apiKeyOptional}
                className="w-full"
              />
              {option?.auth.credentialHelp && (
                <p className="text-xs text-tertiary-alt">
                  {renderHelperText(option.auth.credentialHelp)}
                </p>
              )}
              {state.errors.api_key && (
                <p className="text-xs text-red-500">{state.errors.api_key}</p>
              )}
              {credentialSecretName && (
                <SaveAsSecretToggle
                  fieldKey={credentialSecretName}
                  checked={state.savedAsSecret.api_key ?? false}
                  onToggle={(v) => toggleSecret("api_key", v)}
                />
              )}
            </div>
          ) : null}
        </>
      );
    }

    if (template?.kind !== "stdio") return null;
    const stdio = template;
    return (
      <>
        <SettingsInput
          testId="mcp-install-field-command-readonly"
          name="command-readonly"
          type="text"
          label={t(I18nKey.MCP$COMMAND_LABEL)}
          value={`${stdio.command} ${stdio.args.join(" ")}`.trim()}
          onChange={() => {}}
          isDisabled
          className="w-full"
        />
        {(stdio.envFields ?? []).map((field) => (
          <div key={field.key} className="flex flex-col gap-1">
            <SettingsInput
              testId={`mcp-install-field-${field.key}`}
              name={field.key}
              type={field.type === "password" ? "password" : "text"}
              label={field.label}
              value={state.values[field.key] ?? ""}
              onChange={(v) => setValue(field.key, v)}
              placeholder={field.placeholder}
              required={field.required}
              showOptionalTag={!field.required}
              className="w-full"
            />
            {field.helperText && (
              <p className="text-xs text-tertiary-alt">
                {renderHelperText(field.helperText)}
              </p>
            )}
            {state.errors[field.key] && (
              <p className="text-xs text-red-500">{state.errors[field.key]}</p>
            )}
            {field.key in state.savedAsSecret && (
              <SaveAsSecretToggle
                fieldKey={field.key}
                checked={state.savedAsSecret[field.key]}
                onToggle={(v) => toggleSecret(field.key, v)}
              />
            )}
          </div>
        ))}
        {/* argFields are CLI arguments, not credentials — they don't need
            a "save as secret" toggle and are excluded from savedAsSecret. */}
        {(stdio.argFields ?? []).map((field) => (
          <div key={field.key} className="flex flex-col gap-1">
            <SettingsInput
              testId={`mcp-install-field-${field.key}`}
              name={field.key}
              type={field.type === "password" ? "password" : "text"}
              label={field.label}
              value={state.values[field.key] ?? ""}
              onChange={(v) => setValue(field.key, v)}
              placeholder={field.placeholder}
              required={field.required}
              showOptionalTag={!field.required}
              className="w-full"
            />
            {field.helperText && (
              <p className="text-xs text-tertiary-alt">
                {renderHelperText(field.helperText)}
              </p>
            )}
            {state.errors[field.key] && (
              <p className="text-xs text-red-500">{state.errors[field.key]}</p>
            )}
          </div>
        ))}
      </>
    );
  };

  return (
    <ModalBackdrop onClose={onClose} aria-label={entry.name}>
      <form
        data-testid="mcp-install-modal"
        data-marketplace-id={entry.id}
        onSubmit={handleSubmit}
        className="relative bg-base-secondary p-6 rounded-xl flex flex-col gap-4 border border-[var(--oh-border)] w-[520px] max-w-[90vw] max-h-[85vh] overflow-y-auto custom-scrollbar"
      >
        <ModalCloseButton
          onClose={onClose}
          testId="mcp-install-modal-close"
          disabled={isPending}
        />
        <div className="flex items-start gap-3 pr-6">
          <McpLogoBadge entry={entry} />
          <div className="flex flex-col flex-1">
            <h2 className={modalTitleLgClassName}>{entry.name}</h2>
            <p className="text-xs text-tertiary-light">{entry.description}</p>
          </div>
        </div>

        {entry.installHint && (
          <p className="text-xs text-tertiary-light">{entry.installHint}</p>
        )}

        {entry.docsUrl && (
          <a
            href={entry.docsUrl}
            target="_blank"
            rel="noreferrer"
            className="text-xs text-[var(--oh-muted)] hover:text-white hover:underline self-start transition-colors"
          >
            {t(I18nKey.MCP$VIEW_DOCS)}
          </a>
        )}

        <div className="flex flex-col gap-3">{renderFields()}</div>

        {globalError && (
          <p
            data-testid="mcp-install-modal-error"
            className="text-sm text-red-500 whitespace-pre-wrap"
          >
            {globalError}
          </p>
        )}

        <div className="flex justify-end gap-2 mt-2">
          <BrandButton
            type="button"
            variant="secondary"
            onClick={onClose}
            testId="mcp-install-cancel"
            isDisabled={isPending}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          <BrandButton
            type="submit"
            variant="primary"
            isDisabled={isPending}
            testId="mcp-install-submit"
          >
            {isTesting || isAuthorizingOAuth
              ? t(I18nKey.MCP$VERIFYING)
              : isAdding || isFinalizingInstall
                ? t(I18nKey.SETTINGS$SAVING)
                : t(I18nKey.MCP$INSTALL_BUTTON)}
          </BrandButton>
        </div>
      </form>
    </ModalBackdrop>
  );
}
