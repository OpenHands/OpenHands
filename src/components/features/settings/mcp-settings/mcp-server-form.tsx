import React from "react";
import { useTranslation } from "react-i18next";
import { Trash2 } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { SettingsInput } from "../settings-input";
import { SettingsDropdownInput } from "../settings-dropdown-input";
import { BrandButton } from "../brand-button";
import { OptionalTag } from "../optional-tag";
import { cn } from "#/utils/utils";
import { formControlMultilineFieldClassName } from "#/utils/form-control-classes";
import {
  isValidMcpServerName,
  MCP_SERVER_NAME_PATTERN,
} from "#/utils/mcp-server-name";
import type {
  MCPAuthCredential,
  MCPAuthenticationConfig,
  MCPOAuthClientAuthMethod,
} from "#/types/mcp-auth";
import type { MCPServerConfig } from "#/types/mcp-server";

type MCPServerType = "sse" | "stdio" | "shttp";
type RemoteAuthMode = "none" | "bearer" | "header" | "oauth2";
type OAuthClientAuthMethodOption = "auto" | MCPOAuthClientAuthMethod;

export interface TestMessage {
  ok: boolean;
  text: string;
}

interface MCPServerFormProps {
  mode: "add" | "edit";
  server?: MCPServerConfig;
  existingServers?: MCPServerConfig[];
  onSubmit: (server: MCPServerConfig) => void;
  onCancel: () => void;
  onDelete?: () => void;
  isActionDisabled?: boolean;
  onTest?: (server: MCPServerConfig) => void;
  isTestPending?: boolean;
  testMessage?: TestMessage | null;
}

export function MCPServerForm({
  mode,
  server,
  existingServers,
  onSubmit,
  onCancel,
  onDelete,
  isActionDisabled = false,
  onTest,
  isTestPending = false,
  testMessage = null,
}: MCPServerFormProps) {
  const { t } = useTranslation("openhands");
  const [serverType, setServerType] = React.useState<MCPServerType>(
    server?.type || "sse",
  );
  const [authMode, setAuthMode] = React.useState<RemoteAuthMode>(() => {
    if (server?.auth?.strategy === "oauth2") return "oauth2";
    if (server?.auth?.strategy === "header") return "header";
    if (
      server?.auth?.strategy === "bearer" ||
      server?.auth?.strategy === "api_key"
    ) {
      return "bearer";
    }
    return "none";
  });
  const [oauthClientAuthMethod, setOAuthClientAuthMethod] =
    React.useState<OAuthClientAuthMethodOption>(() =>
      server?.auth?.strategy === "oauth2"
        ? (server.auth.authentication?.client_auth_method ?? "auto")
        : "auto",
    );
  const [error, setError] = React.useState<string | null>(null);
  const formRef = React.useRef<HTMLFormElement>(null);

  const serverTypeOptions = [
    { key: "sse", label: t(I18nKey.SETTINGS$MCP_SERVER_TYPE_SSE) },
    { key: "stdio", label: t(I18nKey.SETTINGS$MCP_SERVER_TYPE_STDIO) },
    { key: "shttp", label: t(I18nKey.SETTINGS$MCP_SERVER_TYPE_SHTTP) },
  ];
  const authModeOptions = [
    { key: "none", label: t(I18nKey.SETTINGS$MCP_AUTH_MODE_NONE) },
    { key: "bearer", label: t(I18nKey.SETTINGS$MCP_AUTH_MODE_BEARER) },
    { key: "header", label: t(I18nKey.SETTINGS$MCP_AUTH_MODE_HEADER) },
    { key: "oauth2", label: t(I18nKey.SETTINGS$MCP_AUTH_MODE_OAUTH) },
  ];
  const oauthClientAuthMethodOptions = [
    { key: "auto", label: t(I18nKey.SETTINGS$MCP_OAUTH_CLIENT_AUTH_AUTO) },
    { key: "none", label: t(I18nKey.SETTINGS$MCP_OAUTH_CLIENT_AUTH_NONE) },
    {
      key: "client_secret_post",
      label: t(I18nKey.SETTINGS$MCP_OAUTH_CLIENT_AUTH_SECRET_POST),
    },
    {
      key: "client_secret_basic",
      label: t(I18nKey.SETTINGS$MCP_OAUTH_CLIENT_AUTH_SECRET_BASIC),
    },
    {
      key: "private_key_jwt",
      label: t(I18nKey.SETTINGS$MCP_OAUTH_CLIENT_AUTH_PRIVATE_KEY_JWT),
    },
  ];

  const validateUrl = (url: string): string | null => {
    if (!url) return t(I18nKey.SETTINGS$MCP_ERROR_URL_REQUIRED);
    try {
      const urlObj = new URL(url);
      if (!["http:", "https:"].includes(urlObj.protocol)) {
        return t(I18nKey.SETTINGS$MCP_ERROR_URL_INVALID_PROTOCOL);
      }
    } catch {
      return t(I18nKey.SETTINGS$MCP_ERROR_URL_INVALID);
    }
    return null;
  };

  const validateName = (name: string): string | null => {
    if (!name) return t(I18nKey.SETTINGS$MCP_ERROR_NAME_REQUIRED);
    if (!isValidMcpServerName(name)) {
      return t(I18nKey.SETTINGS$MCP_ERROR_NAME_INVALID);
    }
    return null;
  };

  const validateNameUniqueness = (name: string): string | null => {
    if (!existingServers) return null;
    const shouldCheckUniqueness =
      mode === "add" || (mode === "edit" && server?.name !== name);
    if (!shouldCheckUniqueness) return null;

    const existingStdioNames = existingServers
      .filter((s) => s.type === "stdio")
      .map((s) => s.name)
      .filter(Boolean);
    if (existingStdioNames.includes(name)) {
      return t(I18nKey.SETTINGS$MCP_ERROR_NAME_DUPLICATE);
    }
    return null;
  };

  const validateCommand = (command: string): string | null => {
    if (!command) return t(I18nKey.SETTINGS$MCP_ERROR_COMMAND_REQUIRED);
    if (command.includes(" ")) {
      return t(I18nKey.SETTINGS$MCP_ERROR_COMMAND_NO_SPACES);
    }
    return null;
  };

  const validateUrlUniqueness = (url: string): string | null => {
    if (!existingServers) return null;
    const originalUrl = server?.url;
    const changed = mode === "add" || (mode === "edit" && originalUrl !== url);
    if (!changed) return null;
    // For URL-based servers (sse/shttp), ensure URL is unique across both types
    const exists = existingServers.some(
      (s) => (s.type === "sse" || s.type === "shttp") && s.url === url,
    );
    if (exists) return t(I18nKey.SETTINGS$MCP_ERROR_URL_DUPLICATE);
    return null;
  };

  const validateEnvFormat = (envString: string): string | null => {
    if (!envString.trim()) return null;
    const lines = envString.split("\n");
    for (let i = 0; i < lines.length; i += 1) {
      const trimmed = lines[i].trim();
      if (trimmed) {
        const eq = trimmed.indexOf("=");
        if (eq === -1) return t(I18nKey.SETTINGS$MCP_ERROR_ENV_INVALID_FORMAT);
        const key = trimmed.substring(0, eq).trim();
        if (!key) return t(I18nKey.SETTINGS$MCP_ERROR_ENV_INVALID_FORMAT);
      }
    }
    return null;
  };

  const validateRemoteAuth = (formData: FormData): string | null => {
    if (authMode === "header") {
      const headerString = formData.get("headers")?.toString() || "";
      if (!headerString.trim())
        return t(I18nKey.SETTINGS$MCP_ERROR_HEADER_REQUIRED);
      return validateEnvFormat(headerString);
    }
    if (authMode === "oauth2") {
      const clientId = formData.get("oauth_client_id")?.toString().trim();
      const clientSecret = formData
        .get("oauth_client_secret")
        ?.toString()
        .trim();
      if (clientSecret && !clientId) {
        return t(I18nKey.SETTINGS$MCP_ERROR_OAUTH_SECRET_REQUIRES_ID);
      }
    }
    return null;
  };

  const validateTimeout = (timeoutStr: string): string | null => {
    if (!timeoutStr.trim()) return null; // Optional field

    const timeout = parseInt(timeoutStr.trim(), 10);
    if (Number.isNaN(timeout)) {
      return t(I18nKey.SETTINGS$MCP_ERROR_TIMEOUT_INVALID_NUMBER);
    }
    if (timeout <= 0) {
      return t(I18nKey.SETTINGS$MCP_ERROR_TIMEOUT_POSITIVE);
    }
    if (timeout > 3600) {
      return t(I18nKey.SETTINGS$MCP_ERROR_TIMEOUT_MAX_EXCEEDED);
    }
    return null;
  };

  const validateStdioServer = (formData: FormData): string | null => {
    const name = formData.get("name")?.toString().trim() || "";
    const command = formData.get("command")?.toString().trim() || "";
    const envString = formData.get("env")?.toString() || "";

    const nameError = validateName(name);
    if (nameError) return nameError;

    const uniquenessError = validateNameUniqueness(name);
    if (uniquenessError) return uniquenessError;

    const commandError = validateCommand(command);
    if (commandError) return commandError;

    // Validate environment variable format
    const envError = validateEnvFormat(envString);
    if (envError) return envError;

    return null;
  };

  const validateForm = (formData: FormData): string | null => {
    if (serverType === "sse" || serverType === "shttp") {
      const url = formData.get("url")?.toString().trim() || "";
      const urlError = validateUrl(url);
      if (urlError) return urlError;
      const urlDupError = validateUrlUniqueness(url);
      if (urlDupError) return urlDupError;

      // The name is optional, but when provided it becomes the mcp_config
      // key (and the reference used in mcp_server_refs), so hold it to the
      // same safe-identifier rule as stdio names.
      const name = formData.get("name")?.toString().trim() || "";
      if (name && !isValidMcpServerName(name)) {
        return t(I18nKey.SETTINGS$MCP_ERROR_NAME_INVALID);
      }

      // Validate timeout for SHTTP servers only
      if (serverType === "shttp") {
        const timeoutStr = formData.get("timeout")?.toString() || "";
        const timeoutError = validateTimeout(timeoutStr);
        if (timeoutError) return timeoutError;
      }

      return validateRemoteAuth(formData);
    }

    if (serverType === "stdio") {
      return validateStdioServer(formData);
    }

    return null;
  };

  const parseEnvironmentVariables = (
    envString: string,
  ): Record<string, string> => {
    const env: Record<string, string> = {};
    const input = envString.trim();
    if (!input) return env;

    for (const line of input.split("\n")) {
      const trimmed = line.trim();
      const eq = trimmed.indexOf("=");
      const key = eq >= 0 ? trimmed.substring(0, eq).trim() : "";
      if (trimmed && eq !== -1 && key) {
        env[key] = trimmed.substring(eq + 1).trim();
      }
    }
    return env;
  };

  const formatEnvironmentVariables = (
    env?: Record<string, string> | null,
  ): string => {
    if (!env) return "";
    return Object.entries(env)
      .map(([key, value]) => `${key}=${value}`)
      .join("\n");
  };

  const editableAuthValue = (auth: MCPAuthCredential | undefined): string => {
    if (auth?.strategy === "bearer" || auth?.strategy === "api_key") {
      return auth.value ?? "";
    }
    return "";
  };

  const editableHeaderValue = (auth: MCPAuthCredential | undefined): string => {
    if (auth?.strategy !== "header") return "";
    return formatEnvironmentVariables(auth.headers);
  };

  const oauthAuthentication =
    server?.auth?.strategy === "oauth2"
      ? server.auth.authentication
      : undefined;
  const oauthState =
    server?.auth?.strategy === "oauth2" ? server.auth.state : undefined;

  const authFromFormData = (
    formData: FormData,
  ): MCPAuthCredential | undefined => {
    if (authMode === "none") return undefined;
    if (authMode === "bearer") {
      const value = formData.get("api_key")?.toString().trim();
      if (!value) return undefined;
      if (server?.auth?.strategy === "api_key") {
        return { ...server.auth, value };
      }
      return { strategy: "bearer", value };
    }
    if (authMode === "header") {
      const headers = parseEnvironmentVariables(
        formData.get("headers")?.toString() || "",
      );
      return { strategy: "header", headers };
    }

    const scopes = formData.get("oauth_scopes")?.toString().trim();
    const clientId = formData.get("oauth_client_id")?.toString().trim();
    const clientSecret = formData.get("oauth_client_secret")?.toString().trim();
    const authentication: MCPAuthenticationConfig = {
      type: "oauth",
      ...(oauthClientAuthMethod !== "auto" && {
        client_auth_method: oauthClientAuthMethod,
      }),
      ...(scopes && { scopes }),
      ...(clientId && { client_id: clientId }),
      ...(clientSecret && { client_secret: clientSecret }),
    };
    return {
      strategy: "oauth2",
      authentication,
      ...(oauthState && { state: oauthState }),
    };
  };

  const buildConfig = (formData: FormData): MCPServerConfig => {
    const baseConfig = {
      id: server?.id || `${serverType}-${Date.now()}`,
      type: serverType,
    };

    if (serverType === "sse" || serverType === "shttp") {
      const name = formData.get("name")?.toString().trim();
      const url = formData.get("url")?.toString().trim();
      const timeoutStr = formData.get("timeout")?.toString().trim();
      const auth = authFromFormData(formData);

      const serverConfig: MCPServerConfig = {
        ...baseConfig,
        ...(name && { name }),
        url: url!,
        ...(auth && { auth }),
      };

      // Only add timeout for SHTTP servers
      if (serverType === "shttp" && timeoutStr) {
        const timeoutValue = parseInt(timeoutStr, 10);
        if (!Number.isNaN(timeoutValue)) {
          serverConfig.timeout = timeoutValue;
        }
      }

      return serverConfig;
    }

    // stdio
    const name = formData.get("name")?.toString().trim();
    const command = formData.get("command")?.toString().trim();
    const argsString = formData.get("args")?.toString().trim();
    const envString = formData.get("env")?.toString().trim();

    const args = argsString
      ? argsString
          .split("\n")
          .map((arg) => arg.trim())
          .filter(Boolean)
      : [];
    const env = parseEnvironmentVariables(envString || "");

    return {
      ...baseConfig,
      name: name!,
      command: command!,
      ...(args.length > 0 && { args }),
      ...(Object.keys(env).length > 0 && { env }),
    };
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    const formData = new FormData(event.currentTarget);
    const validationError = validateForm(formData);

    if (validationError) {
      setError(validationError);
      return;
    }

    onSubmit(buildConfig(formData));
  };

  const handleTestClick = () => {
    if (!onTest || !formRef.current) return;
    setError(null);
    const formData = new FormData(formRef.current);
    const validationError = validateForm(formData);
    if (validationError) {
      setError(validationError);
      return;
    }
    onTest(buildConfig(formData));
  };

  const formTestId =
    mode === "add" ? "add-mcp-server-form" : "edit-mcp-server-form";

  return (
    <form
      ref={formRef}
      data-testid={formTestId}
      onSubmit={handleSubmit}
      className="flex flex-col items-start gap-6"
      noValidate
    >
      {mode === "add" && (
        <SettingsDropdownInput
          testId="server-type-dropdown"
          name="server-type"
          label={t(I18nKey.SETTINGS$MCP_SERVER_TYPE)}
          items={serverTypeOptions}
          selectedKey={serverType}
          onSelectionChange={(key) => setServerType(key as MCPServerType)}
          onInputChange={() => {}} // Prevent input changes
          isClearable={false}
          allowsCustomValue={false}
          required
          wrapperClassName="w-full min-w-0"
        />
      )}

      {error && <p className="text-red-500 text-sm">{error}</p>}

      {(serverType === "sse" || serverType === "shttp") && (
        <>
          <SettingsInput
            testId="server-name-input"
            name="name"
            type="text"
            label={t(I18nKey.SETTINGS$MCP_SERVER_NAME)}
            className="w-full min-w-0"
            showOptionalTag
            defaultValue={server?.name || ""}
            // eslint-disable-next-line i18next/no-literal-string -- example value, not translatable
            placeholder="my_search_server"
            pattern={MCP_SERVER_NAME_PATTERN.source}
          />

          <SettingsInput
            testId="url-input"
            name="url"
            type="url"
            label={t(I18nKey.SETTINGS$MCP_URL)}
            className="w-full min-w-0"
            required
            defaultValue={server?.url || ""}
            // eslint-disable-next-line i18next/no-literal-string -- example value, not translatable
            placeholder="https://api.example.com"
          />

          <SettingsDropdownInput
            testId="auth-mode-dropdown"
            name="auth-mode"
            label={t(I18nKey.SETTINGS$MCP_AUTHENTICATION)}
            items={authModeOptions}
            selectedKey={authMode}
            onSelectionChange={(key) => setAuthMode(key as RemoteAuthMode)}
            onInputChange={() => {}}
            isClearable={false}
            allowsCustomValue={false}
            wrapperClassName="w-full min-w-0"
          />

          {authMode === "bearer" && (
            <SettingsInput
              testId="api-key-input"
              name="api_key"
              type="password"
              label={t(I18nKey.SETTINGS$MCP_API_KEY)}
              className="w-full min-w-0"
              required
              defaultValue={editableAuthValue(server?.auth)}
              placeholder={t(I18nKey.SETTINGS$MCP_API_KEY_PLACEHOLDER)}
            />
          )}

          {authMode === "header" && (
            <label className="flex flex-col gap-2.5 w-full min-w-0">
              <span className="text-sm">{t(I18nKey.SETTINGS$MCP_HEADERS)}</span>
              <textarea
                data-testid="headers-input"
                name="headers"
                rows={4}
                defaultValue={editableHeaderValue(server?.auth)}
                placeholder={t(I18nKey.SETTINGS$MCP_HEADERS_PLACEHOLDER)}
                className={cn(
                  formControlMultilineFieldClassName,
                  "resize-none placeholder:italic",
                  "disabled:bg-[var(--oh-surface-raised)] disabled:border-[var(--oh-border-subtle)]",
                )}
              />
            </label>
          )}

          {authMode === "oauth2" && (
            <>
              <SettingsDropdownInput
                testId="oauth-client-auth-method-dropdown"
                name="oauth_client_auth_method"
                label={t(I18nKey.SETTINGS$MCP_OAUTH_CLIENT_AUTH)}
                items={oauthClientAuthMethodOptions}
                selectedKey={oauthClientAuthMethod}
                onSelectionChange={(key) =>
                  setOAuthClientAuthMethod(key as OAuthClientAuthMethodOption)
                }
                onInputChange={() => {}}
                isClearable={false}
                allowsCustomValue={false}
                wrapperClassName="w-full min-w-0"
              />
              <SettingsInput
                testId="oauth-client-id-input"
                name="oauth_client_id"
                type="text"
                label={t(I18nKey.SETTINGS$MCP_OAUTH_CLIENT_ID)}
                className="w-full min-w-0"
                showOptionalTag
                defaultValue={oauthAuthentication?.client_id || ""}
                placeholder={t(
                  I18nKey.SETTINGS$MCP_OAUTH_CLIENT_ID_PLACEHOLDER,
                )}
              />
              <SettingsInput
                testId="oauth-client-secret-input"
                name="oauth_client_secret"
                type="password"
                label={t(I18nKey.SETTINGS$MCP_OAUTH_CLIENT_SECRET)}
                className="w-full min-w-0"
                showOptionalTag
                defaultValue={oauthAuthentication?.client_secret || ""}
                placeholder={t(
                  I18nKey.SETTINGS$MCP_OAUTH_CLIENT_SECRET_PLACEHOLDER,
                )}
              />
              <SettingsInput
                testId="oauth-scopes-input"
                name="oauth_scopes"
                type="text"
                label={t(I18nKey.SETTINGS$MCP_OAUTH_SCOPES)}
                className="w-full min-w-0"
                showOptionalTag
                defaultValue={
                  Array.isArray(oauthAuthentication?.scopes)
                    ? oauthAuthentication.scopes.join(" ")
                    : oauthAuthentication?.scopes || ""
                }
                placeholder={t(I18nKey.SETTINGS$MCP_OAUTH_SCOPES_PLACEHOLDER)}
              />
            </>
          )}

          {serverType === "shttp" && (
            <SettingsInput
              testId="timeout-input"
              name="timeout"
              type="number"
              label={t(I18nKey.SETTINGS$MCP_TIMEOUT_LABEL)}
              className="w-full min-w-0"
              showOptionalTag
              defaultValue={server?.timeout?.toString() || ""}
              placeholder="60"
              min={1}
              max={3600}
            />
          )}
        </>
      )}

      {serverType === "stdio" && (
        <>
          <SettingsInput
            testId="name-input"
            name="name"
            type="text"
            label={t(I18nKey.SETTINGS$MCP_NAME)}
            className="w-full min-w-0"
            required
            defaultValue={server?.name || ""}
            // eslint-disable-next-line i18next/no-literal-string -- example value, not translatable
            placeholder="my_mcp_server"
            pattern={MCP_SERVER_NAME_PATTERN.source}
          />

          <SettingsInput
            testId="command-input"
            name="command"
            type="text"
            label={t(I18nKey.SETTINGS$MCP_COMMAND)}
            className="w-full min-w-0"
            required
            defaultValue={server?.command || ""}
            // eslint-disable-next-line i18next/no-literal-string -- example value, not translatable
            placeholder="npx"
          />

          <label className="flex flex-col gap-2.5 w-full min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm">
                {t(I18nKey.SETTINGS$MCP_COMMAND_ARGUMENTS)}
              </span>
              <OptionalTag />
            </div>
            <textarea
              data-testid="args-input"
              name="args"
              rows={3}
              defaultValue={server?.args?.join("\n") || ""}
              // eslint-disable-next-line i18next/no-literal-string -- example value, not translatable
              placeholder="arg1&#10;arg2&#10;arg3"
              className={cn(
                formControlMultilineFieldClassName,
                "resize-none placeholder:italic",
                "disabled:bg-[var(--oh-surface-raised)] disabled:border-[var(--oh-border-subtle)]",
              )}
            />
            <p className="text-xs text-tertiary-alt">
              {t(I18nKey.SETTINGS$MCP_COMMAND_ARGUMENTS_HELP)}
            </p>
          </label>

          <label className="flex flex-col gap-2.5 w-full min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm">
                {t(I18nKey.SETTINGS$MCP_ENVIRONMENT_VARIABLES)}
              </span>
              <OptionalTag />
            </div>
            <textarea
              data-testid="env-input"
              name="env"
              rows={4}
              defaultValue={formatEnvironmentVariables(server?.env)}
              // eslint-disable-next-line i18next/no-literal-string -- example value, not translatable
              placeholder="KEY1=value1&#10;KEY2=value2"
              className={cn(
                formControlMultilineFieldClassName,
                "resize-none placeholder:italic",
                "disabled:bg-[var(--oh-surface-raised)] disabled:border-[var(--oh-border-subtle)]",
              )}
            />
          </label>
        </>
      )}

      {testMessage && (
        <p
          data-testid="mcp-test-message"
          className={
            testMessage.ok
              ? "text-sm text-green-500 whitespace-pre-wrap"
              : "text-sm text-red-500 whitespace-pre-wrap"
          }
        >
          {testMessage.text}
        </p>
      )}

      <div
        className={cn(
          "flex w-full items-center gap-2",
          onDelete ? "justify-between" : "justify-end",
        )}
      >
        {onDelete ? (
          <BrandButton
            testId="mcp-custom-editor-delete"
            type="button"
            variant="secondary"
            onClick={onDelete}
            isDisabled={isActionDisabled}
            startContent={
              <Trash2 aria-hidden className="size-4" strokeWidth={2} />
            }
          >
            {t(I18nKey.BUTTON$DELETE)}
          </BrandButton>
        ) : null}
        <div className="flex items-center gap-2">
          <BrandButton
            testId="cancel-button"
            type="button"
            variant="secondary"
            onClick={onCancel}
            isDisabled={isActionDisabled}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          {onTest && (
            <BrandButton
              testId="mcp-test-connection"
              type="button"
              variant="secondary"
              onClick={handleTestClick}
              isDisabled={isActionDisabled || isTestPending}
            >
              {isTestPending
                ? t(I18nKey.MCP$VERIFYING)
                : t(I18nKey.MCP$TEST_BUTTON)}
            </BrandButton>
          )}
          <BrandButton
            testId="submit-button"
            type="submit"
            variant="primary"
            isDisabled={isActionDisabled || isTestPending}
          >
            {mode === "add" && t(I18nKey.SETTINGS$MCP_ADD_SERVER)}
            {mode === "edit" && t(I18nKey.SETTINGS$MCP_SAVE_SERVER)}
          </BrandButton>
        </div>
      </div>
    </form>
  );
}
