import React from "react";
import { useTranslation } from "react-i18next";
import { CircleCheck } from "lucide-react";
import type { IntegrationCatalogEntry as MarketplaceEntry } from "@openhands/extensions/integrations";
import { BrandButton } from "#/components/features/settings/brand-button";
import { BrandBadge } from "#/components/shared/badge";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useSettings } from "#/hooks/query/use-settings";
import type { NativeGitIntegration } from "#/hooks/query/use-native-git-integrations";
import { I18nKey } from "#/i18n/declaration";
import { cloudIntegrationsUrl } from "#/utils/cloud-integrations-url";
import { convertRawProvidersToList } from "#/utils/convert-raw-providers-to-list";

interface NativeIntegrationPanelProps {
  entry: MarketplaceEntry;
  native: NativeGitIntegration;
  /** Fired once the native integration is connected and the user continues. */
  onConnected: () => void;
}

/**
 * The "Native" option for a git integration on a cloud backend. Connecting
 * happens on the instance's Settings > Integrations page (an OAuth flow the
 * canvas cannot host), so Install opens it in a new tab and Continue re-reads
 * the user's connected providers.
 */
export function NativeIntegrationPanel({
  entry,
  native,
  onConnected,
}: NativeIntegrationPanelProps) {
  const { t } = useTranslation("openhands");
  const { backend, orgId } = useActiveBackend();
  const { refetch: refetchSettings } = useSettings();
  const [hasOpenedIntegrations, setHasOpenedIntegrations] =
    React.useState(false);
  const [isChecking, setIsChecking] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const features = [t(I18nKey.MCP$NATIVE_FEATURE_REPOS, { name: entry.name })];
  if (native.provider === "github") {
    features.push(t(I18nKey.MCP$NATIVE_FEATURE_MENTIONS_GITHUB));
  } else if (native.provider === "gitlab") {
    features.push(t(I18nKey.MCP$NATIVE_FEATURE_MENTIONS_GITLAB));
  }

  const openIntegrations = () => {
    window.open(
      cloudIntegrationsUrl(backend, orgId),
      "_blank",
      "noopener,noreferrer",
    );
    setHasOpenedIntegrations(true);
  };

  const confirmConnection = async () => {
    setError(null);
    setIsChecking(true);
    const { data } = await refetchSettings();
    setIsChecking(false);
    if (
      convertRawProvidersToList(data?.provider_tokens_set).includes(
        native.provider,
      )
    ) {
      onConnected();
      return;
    }
    setError(t(I18nKey.MCP$NATIVE_NOT_CONNECTED_YET, { name: entry.name }));
  };

  return (
    <div
      data-testid="mcp-native-panel"
      className="flex flex-col gap-4 rounded-xl border border-border p-4"
    >
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-contrast">
            {t(I18nKey.MCP$NATIVE_TAB, { name: entry.name })}
          </h3>
          <BrandBadge className="shrink-0 whitespace-nowrap px-2.5 py-1 text-xs">
            {t(I18nKey.SETTINGS$SKILLS_RECOMMENDED)}
          </BrandBadge>
        </div>
        <p className="text-xs text-tertiary-light">
          {t(I18nKey.MCP$NATIVE_DESCRIPTION, { name: entry.name })}
        </p>
      </div>

      <ul className="flex flex-col gap-2">
        {features.map((feature) => (
          <li
            key={feature}
            className="flex items-center gap-2 text-sm text-contrast"
          >
            <CircleCheck className="size-4 shrink-0 text-success" aria-hidden />
            {feature}
          </li>
        ))}
      </ul>

      <div className="flex items-center justify-between gap-3 border-t border-border pt-4">
        <p className="text-xs text-tertiary-light">
          {native.isConnected
            ? t(I18nKey.MCP$NATIVE_CONNECTED, { name: entry.name })
            : t(I18nKey.MCP$NATIVE_CONNECT_HINT, { name: entry.name })}
        </p>
        <div className="flex shrink-0 gap-2">
          {!native.isConnected && (
            <BrandButton
              type="button"
              variant={hasOpenedIntegrations ? "secondary" : "primary"}
              onClick={openIntegrations}
              testId="mcp-native-install"
            >
              {t(I18nKey.MCP$INSTALL_BUTTON)}
            </BrandButton>
          )}
          {(native.isConnected || hasOpenedIntegrations) && (
            <BrandButton
              type="button"
              variant="primary"
              onClick={native.isConnected ? onConnected : confirmConnection}
              isDisabled={isChecking}
              testId="mcp-native-continue"
            >
              {t(I18nKey.BUTTON$CONTINUE)}
            </BrandButton>
          )}
        </div>
      </div>

      {error && (
        <p data-testid="mcp-native-error" className="text-sm text-red-500">
          {error}
        </p>
      )}
    </div>
  );
}
