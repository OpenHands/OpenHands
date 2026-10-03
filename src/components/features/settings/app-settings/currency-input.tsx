import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { COST_CURRENCIES } from "#/utils/cost-currency";
import { useCostCurrencyStore } from "#/stores/cost-currency-store";
import { SettingsDropdownInput } from "../settings-dropdown-input";

export function CurrencyInput() {
  const { t } = useTranslation("openhands");
  const currency = useCostCurrencyStore((state) => state.currency);
  const setCurrency = useCostCurrencyStore((state) => state.setCurrency);

  return (
    <div className="flex flex-col gap-2">
      <SettingsDropdownInput
        testId="cost-currency-input"
        name="cost-currency-input"
        label={t(I18nKey.SETTINGS$COST_CURRENCY)}
        items={COST_CURRENCIES.map((code) => ({
          key: code,
          label: code,
        }))}
        selectedKey={currency}
        onSelectionChange={(key) => {
          if (key) setCurrency(String(key));
        }}
        isClearable={false}
        wrapperClassName="w-full min-w-0"
      />
      <p className="text-sm leading-5 text-tertiary-light">
        {t(I18nKey.SETTINGS$COST_CURRENCY_HINT)}
      </p>
    </div>
  );
}
