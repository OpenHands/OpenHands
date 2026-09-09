import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "#/ui/toggle-switch";
import { I18nKey } from "#/i18n/declaration";
import { mapProvider } from "#/utils/map-provider";
import { settingsListRowClassName } from "#/utils/settings-list-classes";
import { cn } from "#/utils/utils";
import type {
  SubscriptionModelOffer,
  SubscriptionSource,
} from "#/utils/subscription-model-catalog";

function sourceLabel(
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

interface SubscriptionModelRowProps {
  offer: SubscriptionModelOffer;
  otherSources: SubscriptionSource[];
  enabled: boolean;
  onToggle?: (enabled: boolean) => void;
}

export function SubscriptionModelRow({
  offer,
  otherSources,
  enabled,
  onToggle,
}: SubscriptionModelRowProps) {
  const { t } = useTranslation("openhands");
  const alsoOn =
    otherSources.length > 0
      ? otherSources.map((source) => sourceLabel(source, t)).join(", ")
      : "";

  return (
    <div
      data-testid="subscription-model-row"
      className={cn(settingsListRowClassName, "justify-between gap-3")}
    >
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span
          className="min-w-0 truncate text-sm font-medium text-white"
          title={offer.id}
        >
          {offer.label}
        </span>
        {alsoOn ? (
          <span
            data-testid="subscription-model-also-on"
            className="truncate text-xs text-[var(--oh-muted)]"
          >
            {t(I18nKey.SETTINGS$SUBSCRIPTION_MODEL_ALSO_ON, {
              sources: alsoOn,
            })}
          </span>
        ) : null}
      </div>
      {onToggle ? (
        <ToggleSwitch
          enabled={enabled}
          label={t(I18nKey.SETTINGS$SUBSCRIPTION_MODEL_TOGGLE, {
            model: offer.label,
          })}
          onToggle={() => onToggle(!enabled)}
        />
      ) : null}
    </div>
  );
}
