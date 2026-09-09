import { useTranslation } from "react-i18next";
import { ChannelsOverview } from "#/components/features/channels/channels-overview";
import { ChannelsSubpageLayout } from "#/components/features/channels/channels-subpage-layout";
import { I18nKey } from "#/i18n/declaration";

export default function ChannelsRoute() {
  const { t } = useTranslation("openhands");
  return (
    <ChannelsSubpageLayout>
      <div className="min-w-0">
        <h1 className="text-xl font-semibold text-content">
          {t(I18nKey.CHANNELS$CONNECTIONS)}
        </h1>
        <p className="mt-1 text-sm text-muted">
          {t(I18nKey.SETTINGS$PAGE_CHANNELS_SUBLINE)}
        </p>
      </div>
      <ChannelsOverview />
    </ChannelsSubpageLayout>
  );
}
