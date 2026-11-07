import { useTranslation } from "react-i18next";
import { ExternalLink } from "lucide-react";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";

interface McpManagedInSettingsProps {
  /** The instance's Settings > MCP page, opened in a new tab. */
  settingsUrl: string;
}

/**
 * Stands in for the MCP page body when the instance manages MCP servers on
 * its own Settings > MCP page, which also lists the organization's servers
 * (OHE-3869).
 */
export function McpManagedInSettings({
  settingsUrl,
}: McpManagedInSettingsProps) {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="mcp-managed-in-settings"
      className="flex flex-col items-start gap-4 rounded-xl border border-border p-6"
    >
      <div className="space-y-1">
        <h2 className="text-xl font-medium leading-6 text-foreground">
          {t(I18nKey.MCP$MANAGED_IN_SETTINGS_TITLE)}
        </h2>
        <p className="max-w-2xl text-sm text-tertiary-light">
          {t(I18nKey.MCP$MANAGED_IN_SETTINGS_DESCRIPTION)}
        </p>
      </div>
      <BrandButton
        type="button"
        variant="primary"
        testId="mcp-open-settings"
        className="flex items-center gap-2"
        onClick={() =>
          window.open(settingsUrl, "_blank", "noopener,noreferrer")
        }
      >
        {t(I18nKey.MCP$OPEN_MCP_SETTINGS)}
        <ExternalLink className="size-4 shrink-0" aria-hidden />
      </BrandButton>
    </div>
  );
}
