import React from "react";
import { useTranslation } from "react-i18next";
import type { ChannelState } from "#/api/channel-service/channel-types";
import { ChannelConfig } from "#/components/features/channels/channel-config";
import { MeetilyImport } from "#/components/features/channels/meetily";
import { BrandButton } from "#/components/features/settings/brand-button";
import {
  useChannelMessages,
  useChannels,
  useStartChannel,
  useStopChannel,
} from "#/hooks/query/use-channels";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import { cn } from "#/utils/utils";

function stateKey(state: ChannelState): I18nKey {
  if (state === "running") return I18nKey.CHANNELS$RUNNING;
  if (state === "unconfigured") return I18nKey.CHANNELS$UNCONFIGURED;
  if (state === "error") return I18nKey.CHANNELS$ERROR;
  return I18nKey.CHANNELS$STOPPED;
}

export function ChannelsOverview() {
  const { t } = useTranslation("openhands");
  const channelsQuery = useChannels();
  const messagesQuery = useChannelMessages({ limit: 50 });
  const startChannel = useStartChannel();
  const stopChannel = useStopChannel();
  const [configId, setConfigId] = React.useState<string | null>(null);
  const channels = channelsQuery.data ?? [];
  const configChannel = channels.find((channel) => channel.id === configId);

  return (
    <div data-testid="channels-overview" className="flex flex-col gap-4">
      <Typography variant="h1">{t(I18nKey.CHANNELS$TITLE)}</Typography>
      {channels.length === 0 ? (
        <p className="text-sm text-tertiary-light">
          {t(I18nKey.CHANNELS$EMPTY)}
        </p>
      ) : (
        <ul className="flex flex-col gap-2">
          {channels.map((channel) => {
            const running = channel.status.state === "running";
            return (
              <li
                key={channel.id}
                data-testid={`channel-row-${channel.id}`}
                className="flex flex-wrap items-center justify-between gap-2 rounded-xl bg-base-secondary p-3"
              >
                <div>
                  <h2 className="text-sm font-medium text-white">
                    {channel.id}
                  </h2>
                  <span
                    data-testid={`channel-state-${channel.id}`}
                    className={cn(
                      extensionModuleCardPillClassName,
                      "text-white",
                    )}
                  >
                    {t(stateKey(channel.status.state))}
                  </span>
                </div>
                <div className="flex gap-2">
                  <BrandButton
                    type="button"
                    variant="tertiary"
                    testId={`channel-config-${channel.id}`}
                    onClick={() => setConfigId(channel.id)}
                  >
                    {t(I18nKey.CHANNELS$CONFIG)}
                  </BrandButton>
                  <BrandButton
                    type="button"
                    variant="primary"
                    testId={`channel-toggle-${channel.id}`}
                    onClick={() => {
                      if (running) {
                        stopChannel.mutate(channel.id);
                      } else {
                        startChannel.mutate(channel.id);
                      }
                    }}
                  >
                    {running
                      ? t(I18nKey.CHANNELS$STOP)
                      : t(I18nKey.CHANNELS$START)}
                  </BrandButton>
                </div>
              </li>
            );
          })}
        </ul>
      )}
      <section>
        <Typography variant="h2">{t(I18nKey.CHANNELS$MESSAGES)}</Typography>
        {(messagesQuery.data?.items.length ?? 0) === 0 ? (
          <p className="mt-2 text-sm text-tertiary-light">
            {t(I18nKey.CHANNELS$EMPTY_LOG)}
          </p>
        ) : (
          <ul className="mt-2 flex flex-col gap-2">
            {messagesQuery.data?.items.map((message) => (
              <li
                key={message.id}
                data-testid={`channel-message-${message.id}`}
                className="rounded-xl bg-base-secondary p-3 text-sm text-white"
              >
                <span
                  data-testid={`channel-message-direction-${message.id}`}
                  className={cn(extensionModuleCardPillClassName, "mr-2")}
                >
                  {message.direction === "inbound"
                    ? t(I18nKey.CHANNELS$INBOUND)
                    : t(I18nKey.CHANNELS$OUTBOUND)}
                </span>
                <span>
                  {t(I18nKey.CHANNELS$SOURCE)} {message.source}
                </span>
                <span className="ml-2">
                  {t(I18nKey.CHANNELS$THREAD)} {message.thread_ref}
                </span>
                <span className="ml-2">
                  {t(I18nKey.CHANNELS$CORRELATION)} {message.correlation_id}
                </span>
                <p className="mt-1 text-tertiary-light">{message.text}</p>
              </li>
            ))}
          </ul>
        )}
      </section>
      <MeetilyImport />
      {configChannel ? (
        <ChannelConfig
          channel={configChannel}
          isOpen
          onClose={() => setConfigId(null)}
        />
      ) : null}
    </div>
  );
}
