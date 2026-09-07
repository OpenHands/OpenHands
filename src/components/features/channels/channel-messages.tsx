import { useTranslation } from "react-i18next";
import { useChannelMessages } from "#/hooks/query/use-channels";
import { I18nKey } from "#/i18n/declaration";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import { cn } from "#/utils/utils";

export function ChannelMessages() {
  const { t } = useTranslation("openhands");
  const messagesQuery = useChannelMessages({ limit: 50 });

  return (
    <div data-testid="channels-messages" className="flex flex-col gap-4">
      {(messagesQuery.data?.items.length ?? 0) === 0 ? (
        <p className="text-sm text-tertiary-light">
          {t(I18nKey.CHANNELS$EMPTY_LOG)}
        </p>
      ) : (
        <ul className="flex flex-col gap-2">
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
    </div>
  );
}
