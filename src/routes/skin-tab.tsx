import { useTranslation } from "react-i18next";
import { SKIN_APP_BASE } from "#/api/skin-service";
import { useSkinStatus } from "#/hooks/query/use-skin";
import { I18nKey } from "#/i18n/declaration";

/**
 * Default tab of a skinned instance: renders the installed skin's web app
 * (running on OPENHANDS_SKIN_PORT, reverse-proxied verbatim under /skin) in a
 * full-size iframe filling the main content area.
 */
function SkinTab() {
  const { t } = useTranslation("openhands");
  const { data: status, isLoading } = useSkinStatus();

  if (isLoading) {
    return null;
  }

  if (!status?.installed) {
    return (
      <main
        data-testid="skin-tab-empty"
        className="flex h-full items-center justify-center text-sm text-[var(--oh-muted)]"
      >
        {t(I18nKey.SKIN$NOT_INSTALLED)}
      </main>
    );
  }

  return (
    <main data-testid="skin-tab" className="h-full w-full">
      <iframe
        data-testid="skin-iframe"
        title={status.name || t(I18nKey.SKIN$TITLE)}
        src={`${SKIN_APP_BASE}/`}
        className="h-full w-full border-0"
        sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-downloads"
      />
    </main>
  );
}

export default SkinTab;
