import React from "react";
import { RefreshCw } from "lucide-react";
import { useTranslation } from "react-i18next";
import { getAgentServerFormDefaults } from "#/api/agent-server-config";
import {
  getRegisteredBackends,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import {
  SEEDED_DEFAULT_BACKEND_ID,
  DEFAULT_LOCAL_BACKEND_NAME,
} from "#/api/backend-registry/default-backend";
import type { Backend } from "#/api/backend-registry/types";
import { cn } from "#/utils/utils";
import { buildAgentCanvasPath } from "#/utils/base-path";
import { BrandButton } from "./brand-button";
import { SettingsInput } from "./settings-input";
import { I18nKey } from "#/i18n/declaration";

type AgentServerConnectionFormVariant = "settings" | "onboarding";

interface AgentServerConnectionFormProps {
  className?: string;
  formClassName?: string;
  variant?: AgentServerConnectionFormVariant;
  showSectionHeader?: boolean;
}

export function AgentServerConnectionForm({
  className,
  formClassName,
  variant = "onboarding",
  showSectionHeader,
}: AgentServerConnectionFormProps) {
  const { t } = useTranslation("openhands");
  const defaults = React.useMemo(() => getAgentServerFormDefaults(), []);
  const [baseUrl, setBaseUrl] = React.useState(defaults.baseUrl);
  const [sessionApiKey, setSessionApiKey] = React.useState(
    defaults.sessionApiKey,
  );

  const formIsClean =
    baseUrl === defaults.baseUrl && sessionApiKey === defaults.sessionApiKey;
  const isOnboarding = variant === "onboarding";
  const shouldShowSectionHeader = showSectionHeader ?? isOnboarding;

  const reconnect = () => {
    window.location.assign(buildAgentCanvasPath("/"));
  };

  const syncDefaultBackendInRegistry = () => {
    const trimmedHost = baseUrl.trim();
    if (!trimmedHost) return;

    const trimmedKey = sessionApiKey.trim();
    const current = getRegisteredBackends();
    const defaultEntry: Backend = {
      id: SEEDED_DEFAULT_BACKEND_ID,
      name: DEFAULT_LOCAL_BACKEND_NAME,
      host: trimmedHost,
      apiKey: trimmedKey,
      kind: "local",
    };

    const existingIndex = current.findIndex(
      (b) => b.id === SEEDED_DEFAULT_BACKEND_ID,
    );
    if (existingIndex === -1) {
      setRegisteredBackends([defaultEntry, ...current]);
      return;
    }

    const next = current.slice();
    next[existingIndex] = {
      ...current[existingIndex],
      host: trimmedHost,
      apiKey: trimmedKey,
    };
    setRegisteredBackends(next);
  };

  const onSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    syncDefaultBackendInRegistry();

    reconnect();
  };

  return (
    <form
      data-testid="agent-server-connection-form"
      onSubmit={onSubmit}
      className={cn("flex h-full flex-col", className)}
    >
      <div
        className={cn(
          "flex flex-col gap-5",
          isOnboarding &&
            "rounded-3xl border border-white/10 bg-[var(--oh-surface-deep)]/80 p-6 shadow-2xl",
          formClassName,
        )}
      >
        {shouldShowSectionHeader ? (
          <div>
            <p className="text-sm font-medium uppercase tracking-[0.24em] text-primary">
              {t(I18nKey.SETTINGS$AGENT_SERVER_CONNECTION_DETAILS_TITLE)}
            </p>
            <p className="mt-3 w-full min-w-0 text-sm leading-7 text-[var(--oh-muted)]">
              {t(I18nKey.SETTINGS$AGENT_SERVER_CONNECTION_DETAILS_DESCRIPTION)}
            </p>
          </div>
        ) : null}

        <SettingsInput
          testId="agent-server-url-input"
          name="agent-server-url-input"
          type="text"
          label={t(I18nKey.SETTINGS$AGENT_SERVER_URL)}
          value={baseUrl}
          onChange={setBaseUrl}
          placeholder={t(I18nKey.SETTINGS$AGENT_SERVER_URL_PLACEHOLDER)}
          className="w-full min-w-0"
        />

        <SettingsInput
          testId="agent-server-api-key-input"
          name="agent-server-api-key-input"
          type="password"
          label={t(I18nKey.SETTINGS$AGENT_SERVER_API_KEY)}
          value={sessionApiKey}
          onChange={setSessionApiKey}
          placeholder={t(I18nKey.SETTINGS$AGENT_SERVER_API_KEY_PLACEHOLDER)}
          showOptionalTag
          className="w-full min-w-0"
        />

        <p className="w-full min-w-0 text-xs leading-6 text-[var(--oh-text-subtle)]">
          {t(I18nKey.SETTINGS$AGENT_SERVER_BROWSER_ONLY_NOTE)}
        </p>
      </div>

      <div className="mt-6 flex flex-wrap gap-3">
        <BrandButton
          testId="retry-connection-button"
          variant="secondary"
          type="button"
          onClick={reconnect}
          startContent={<RefreshCw className="size-4" />}
        >
          {t(I18nKey.SETTINGS$AGENT_SERVER_RETRY_CONNECTION)}
        </BrandButton>
        <BrandButton
          testId="submit-button"
          variant="primary"
          type="submit"
          isDisabled={formIsClean}
        >
          {t(I18nKey.SETTINGS$SAVE_AND_RECONNECT)}
        </BrandButton>
      </div>
    </form>
  );
}
