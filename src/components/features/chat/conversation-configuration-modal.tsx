import React from "react";
import { Eye, EyeOff, Puzzle } from "lucide-react";
import { useTranslation } from "react-i18next";
import { INTEGRATION_CATALOG } from "@openhands/extensions/integrations";
import {
  changeConversationConfiguration,
  type ConversationConfigurationOperation,
} from "#/api/conversation-configuration-service";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { McpLogoBadge } from "#/components/features/mcp-logo-badge";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import {
  ModalBody,
  MODAL_MAX_WIDTH_VIEWPORT,
} from "#/components/shared/modals/modal-body";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { NavigationLink } from "#/components/shared/navigation-link";
import { useSettings } from "#/hooks/query/use-settings";
import { I18nKey } from "#/i18n/declaration";
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
}: {
  onAttach: (ref: string) => void;
  pending: boolean;
  attached: Set<string>;
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
      <NavigationLink
        to="/mcp"
        target="_blank"
        rel="noopener noreferrer"
        className="text-sm text-primary underline"
      >
        {t(I18nKey.MCP$VIEW_MARKETPLACE)}
      </NavigationLink>
    </section>
  );
}

export function ConversationConfigurationModal({
  conversationId,
  isAcp,
  onClose,
}: ConversationConfigurationModalProps) {
  const { t } = useTranslation("openhands");
  const [name, setName] = React.useState("");
  const [value, setValue] = React.useState("");
  const [showValue, setShowValue] = React.useState(false);
  const [pending, setPending] = React.useState(false);
  const [result, setResult] = React.useState<AppliedChange | null>(null);
  const [attached, setAttached] = React.useState(() => new Set<string>());
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
        setAttached((current) => new Set(current).add(operation.ref));
      }
      setResult({ type: "success", name: label });
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
            attached={attached}
          />
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
