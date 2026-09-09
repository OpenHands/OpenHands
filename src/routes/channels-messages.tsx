import { useTranslation } from "react-i18next";
import { ChannelMessages } from "#/components/features/channels/channel-messages";
import { ChannelsSubpageLayout } from "#/components/features/channels/channels-subpage-layout";
import { I18nKey } from "#/i18n/declaration";

export default function ChannelsMessagesRoute() {
  const { t } = useTranslation("openhands");
  return (
    <ChannelsSubpageLayout>
      <div className="min-w-0">
        <h1 className="text-xl font-semibold text-content">
          {t(I18nKey.CHANNELS$MESSAGES)}
        </h1>
      </div>
      <ChannelMessages />
    </ChannelsSubpageLayout>
  );
}
