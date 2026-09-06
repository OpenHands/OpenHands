import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

export function HowAutoRoutes() {
  const { t } = useTranslation("openhands");
  return (
    <section data-testid="routing-how-auto" className="flex flex-col gap-2">
      <h3 className="text-sm font-medium text-white">
        {t(I18nKey.ROUTING$HOW_AUTO)}
      </h3>
      <ol className="list-decimal pl-5 text-sm text-[var(--oh-muted)]">
        <li data-testid="routing-precedence-specific">
          {t(I18nKey.ROUTING$PRECEDENCE_SPECIFIC)}
        </li>
        <li data-testid="routing-precedence-sensitivity">
          {t(I18nKey.ROUTING$PRECEDENCE_SENSITIVITY)}
        </li>
        <li data-testid="routing-precedence-default">
          {t(I18nKey.ROUTING$PRECEDENCE_DEFAULT)}
        </li>
      </ol>
      <ol className="list-decimal pl-5 text-sm text-[var(--oh-muted)]">
        <li>{t(I18nKey.ROUTING$ORDER_CLASSIFY)}</li>
        <li>{t(I18nKey.ROUTING$ORDER_FILTER)}</li>
        <li>{t(I18nKey.ROUTING$ORDER_RANK)}</li>
        <li>{t(I18nKey.ROUTING$ORDER_PICK)}</li>
      </ol>
    </section>
  );
}
