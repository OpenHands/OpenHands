import { useTranslation } from "react-i18next";
import { SubscriptionModelRow } from "./subscription-model-row";
import { I18nKey } from "#/i18n/declaration";
import { mapProvider } from "#/utils/map-provider";
import {
  settingsListDividerClassName,
  settingsListScrollContainerClassName,
} from "#/utils/settings-list-classes";
import { cn } from "#/utils/utils";
import {
  otherSourcesForOffer,
  SUBSCRIPTION_SOURCES,
  type MergedSubscriptionModel,
  type SubscriptionModelOffer,
  type SubscriptionSource,
} from "#/utils/subscription-model-catalog";

function sourceHeading(
  source: SubscriptionSource,
  t: (key: I18nKey) => string,
): string {
  if (source === "chatgpt")
    return t(I18nKey.SETTINGS$LLM_AUTH_TYPE_SUBSCRIPTION);
  if (source === "claude") {
    return t(I18nKey.SETTINGS$LLM_AUTH_TYPE_CLAUDE_SUBSCRIPTION);
  }
  return mapProvider(source);
}

interface SubscriptionModelsBodyProps {
  offers: SubscriptionModelOffer[];
  rows: MergedSubscriptionModel[];
  isOfferEnabled: (offer: SubscriptionModelOffer) => boolean;
  onToggle: (offer: SubscriptionModelOffer, enabled: boolean) => void;
  canManage?: boolean;
}

export function SubscriptionModelsBody({
  offers,
  rows,
  isOfferEnabled,
  onToggle,
  canManage = true,
}: SubscriptionModelsBodyProps) {
  const { t } = useTranslation("openhands");

  return (
    <div className="flex flex-col gap-4" data-testid="subscription-models-body">
      {SUBSCRIPTION_SOURCES.map((source) => {
        const sourceOffers = offers.filter((offer) => offer.source === source);
        if (sourceOffers.length === 0) return null;
        return (
          <div key={source} className="flex flex-col gap-2">
            <h3
              data-testid={`subscription-model-group-${source}`}
              className="text-xs font-medium uppercase tracking-wide text-[var(--oh-muted)]"
            >
              {sourceHeading(source, t)}
            </h3>
            <div
              className={cn(
                settingsListScrollContainerClassName,
                settingsListDividerClassName,
              )}
            >
              {sourceOffers.map((offer) => (
                <SubscriptionModelRow
                  key={`${offer.source}:${offer.id}`}
                  offer={offer}
                  otherSources={otherSourcesForOffer(rows, offer)}
                  enabled={isOfferEnabled(offer)}
                  onToggle={
                    canManage
                      ? (enabled) => onToggle(offer, enabled)
                      : undefined
                  }
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
